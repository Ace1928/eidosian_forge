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
def mapRectToDevice(self, rect):
    """
        Return *rect* mapped from local coordinates to device coordinates (pixels).
        If there is no device mapping available, return None.
        """
    vt = self.deviceTransform()
    if vt is None:
        return None
    return vt.mapRect(rect)