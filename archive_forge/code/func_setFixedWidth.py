from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsItem import GraphicsItem
def setFixedWidth(self, h):
    self.setMaximumWidth(h)
    self.setMinimumWidth(h)