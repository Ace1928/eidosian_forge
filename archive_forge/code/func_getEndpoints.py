from math import atan2, degrees
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
def getEndpoints(self):
    if self._endpoints[0] is None:
        lr = self.line.boundingRect()
        pt1 = Point(lr.left(), 0)
        pt2 = Point(lr.right(), 0)
        if self.line.angle % 90 != 0:
            view = self.getViewBox()
            if not self.isVisible() or not isinstance(view, ViewBox):
                return (None, None)
            p = QtGui.QPainterPath()
            p.moveTo(pt1)
            p.lineTo(pt2)
            p = self.line.itemTransform(view)[0].map(p)
            vr = QtGui.QPainterPath()
            vr.addRect(view.boundingRect())
            paths = vr.intersected(p).toSubpathPolygons(QtGui.QTransform())
            if len(paths) > 0:
                l = list(paths[0])
                pt1 = self.line.mapFromItem(view, l[0])
                pt2 = self.line.mapFromItem(view, l[1])
        self._endpoints = (pt1, pt2)
    return self._endpoints