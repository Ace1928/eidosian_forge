import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def getSceneHandlePositions(self, index=None):
    """Returns the position of handles in the scene coordinate system.
        
        The format returned is a list of (name, pos) tuples.
        """
    if index is None:
        positions = []
        for h in self.handles:
            positions.append((h['name'], h['item'].scenePos()))
        return positions
    else:
        return (self.handles[index]['name'], self.handles[index]['item'].scenePos())