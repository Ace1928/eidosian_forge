import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def labelChanged(self):
    newName = self.label.toPlainText()
    if newName != self.term.name():
        self.term.rename(newName)