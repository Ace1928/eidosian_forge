import operator
import weakref
from collections import OrderedDict
from functools import reduce
from math import hypot
from typing import Optional
from xml.etree.ElementTree import Element
from .. import functions as fn
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QtCore, QtWidgets, isQObjectAlive
def pixelVectors(self, direction=None):
    """Return vectors in local coordinates representing the width and height of a view pixel.
        If direction is specified, then return vectors parallel and orthogonal to it.
        
        Return (None, None) if pixel size is not yet defined (usually because the item has not yet been displayed)
        or if pixel size is below floating-point precision limit.
        """
    dt = self.deviceTransform()
    if dt is None:
        return (None, None)
    dt.setMatrix(dt.m11(), dt.m12(), 0, dt.m21(), dt.m22(), 0, 0, 0, 1)
    if direction is None:
        direction = QtCore.QPointF(1, 0)
    elif direction.manhattanLength() == 0:
        raise Exception('Cannot compute pixel length for 0-length vector.')
    key = (dt.m11(), dt.m21(), dt.m12(), dt.m22(), direction.x(), direction.y())
    if key == self._pixelVectorCache[0]:
        return tuple(map(Point, self._pixelVectorCache[1]))
    pv = self._pixelVectorGlobalCache.get(key, None)
    if pv is not None:
        self._pixelVectorCache = [key, pv]
        return tuple(map(Point, pv))
    directionr = direction
    dirLine = QtCore.QLineF(QtCore.QPointF(0, 0), directionr)
    viewDir = dt.map(dirLine)
    if viewDir.length() == 0:
        return (None, None)
    try:
        normView = viewDir.unitVector()
        normOrtho = normView.normalVector()
    except:
        raise Exception('Invalid direction %s' % directionr)
    dti = fn.invertQTransform(dt)
    pv = (Point(dti.map(normView).p2()), Point(dti.map(normOrtho).p2()))
    self._pixelVectorCache[1] = pv
    self._pixelVectorCache[0] = key
    self._pixelVectorGlobalCache[key] = pv
    return self._pixelVectorCache[1]