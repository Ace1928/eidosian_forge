from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
def lockXRange(self, v1):
    if not v1 in self.lockedViewports:
        self.lockedViewports.append(v1)