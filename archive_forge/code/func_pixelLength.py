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
def pixelLength(self, direction, ortho=False):
    """Return the length of one pixel in the direction indicated (in local coordinates)
        If ortho=True, then return the length of one pixel orthogonal to the direction indicated.
        
        Return None if pixel size is not yet defined (usually because the item has not yet been displayed).
        """
    normV, orthoV = self.pixelVectors(direction)
    if normV is None or orthoV is None:
        return None
    if ortho:
        return orthoV.length()
    return normV.length()