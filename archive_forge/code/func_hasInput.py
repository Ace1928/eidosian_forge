import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def hasInput(self):
    for t in self.connections():
        if t.isOutput():
            return True
    return False