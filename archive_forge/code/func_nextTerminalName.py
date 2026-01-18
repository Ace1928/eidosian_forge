import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def nextTerminalName(self, name):
    """Return an unused terminal name"""
    name2 = name
    i = 1
    while name2 in self.terminals:
        name2 = '%s.%d' % (name, i)
        i += 1
    return name2