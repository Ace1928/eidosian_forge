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
def mapRectFromView(self, obj):
    vt = self.viewTransform()
    if vt is None:
        return None
    vt = fn.invertQTransform(vt)
    return vt.mapRect(obj)