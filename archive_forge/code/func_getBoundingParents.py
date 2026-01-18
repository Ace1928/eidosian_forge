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
def getBoundingParents(self):
    """Return a list of parents to this item that have child clipping enabled."""
    p = self
    parents = []
    while True:
        p = p.parentItem()
        if p is None:
            break
        if p.flags() & self.GraphicsItemFlag.ItemClipsChildrenToShape:
            parents.append(p)
    return parents