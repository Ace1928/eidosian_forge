import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def nodeMoved(self):
    for t, item in self.term.connections().items():
        item.updateLine()