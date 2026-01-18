from math import atan2, degrees
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject
def dataBounds(self, ax, frac=1.0, orthoRange=None):
    """
        Returns only the anchor point for when calulating view ranges.
        
        Sacrifices some visual polish for fixing issue #2642.
        """
    if orthoRange:
        range_min, range_max = (orthoRange[0], orthoRange[1])
        if not range_min <= self.anchor[ax] <= range_max:
            return [None, None]
    return [self.anchor[ax], self.anchor[ax]]