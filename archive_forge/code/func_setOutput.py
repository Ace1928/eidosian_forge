import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def setOutput(self, **vals):
    self.setOutputNoSignal(**vals)
    self.sigOutputChanged.emit(self)