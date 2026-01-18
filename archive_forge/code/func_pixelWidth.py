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
def pixelWidth(self):
    vt = self.deviceTransform()
    if vt is None:
        return 0
    vt = fn.invertQTransform(vt)
    return vt.map(QtCore.QLineF(0, 0, 1, 0)).length()