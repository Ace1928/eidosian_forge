import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def setValueAcceptable(self, v=True):
    self.valueOk = v
    self.recolor()