import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def drawPicture(self):
    self.picture = QtGui.QPicture()
    painter = QtGui.QPainter(self.picture)
    self._render(painter)
    painter.end()