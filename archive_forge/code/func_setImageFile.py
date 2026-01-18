from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def setImageFile(self, imageFile):
    self.setPixmap(QtGui.QPixmap(imageFile))