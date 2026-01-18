import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def parentBounds(self):
    """
        Return the bounding rectangle of this ROI in the coordinate system
        of its parent.        
        """
    return self.mapToParent(self.boundingRect()).boundingRect()