from .iq_picture import PictureIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
def setResultPictureProps(self, pictureData, pictureId, preview=True):
    self.preview = preview
    self.pictureData = pictureData
    self.pictureId = pictureId