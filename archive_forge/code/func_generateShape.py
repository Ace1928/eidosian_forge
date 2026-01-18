import string
from math import atan2
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .ScatterPlotItem import Symbols
from .TextItem import TextItem
from .UIGraphicsItem import UIGraphicsItem
from .ViewBox import ViewBox
def generateShape(self):
    dt = self.deviceTransform()
    if dt is None:
        self._shape = self._path
        return None
    v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
    dti = fn.invertQTransform(dt)
    devPos = dt.map(QtCore.QPointF(0, 0))
    tr = QtGui.QTransform()
    tr.translate(devPos.x(), devPos.y())
    va = atan2(v.y(), v.x())
    tr.rotateRadians(va)
    tr.scale(self.scale, self.scale)
    return dti.map(tr.map(self._path))