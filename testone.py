from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/tiny-yolo.cfg", "load": 5125, "threshold": 0.13}

tfnet = TFNet(options)

imgcv = cv2.imread("./test/training/images/20200104.jpeg")
# flow --pbLoad built_graph/tiny-yolo.pb --metaLoad built_graph/tiny-yolo.meta --imgdir test/testing/
result = tfnet.return_predict(imgcv)
print(result)