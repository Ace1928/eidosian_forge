from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def setSpan(self, mn, mx):
    if self.span == (mn, mx):
        return
    self.span = (mn, mx)
    for line in self.lines:
        line.setSpan(mn, mx)
    self.update()