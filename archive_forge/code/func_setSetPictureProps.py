from .iq_picture import PictureIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import time
def setSetPictureProps(self, previewData, pictureData, pictureId=None):
    self.setPictureData(pictureData)
    self.setPictureId(pictureId or str(int(time.time())))
    self.setPreviewData(previewData)