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
def mapFromDevice(self, obj):
    """
        Return *obj* mapped from device coordinates (pixels) to local coordinates.
        If there is no device mapping available, return None.
        """
    vt = self.deviceTransform()
    if vt is None:
        return None
    if isinstance(obj, QtCore.QPoint):
        obj = QtCore.QPointF(obj)
    vt = fn.invertQTransform(vt)
    return vt.map(obj)