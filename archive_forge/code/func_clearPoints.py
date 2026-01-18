import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def clearPoints(self):
    """
        Remove all handles and segments.
        """
    while len(self.handles) > 0:
        self.removeHandle(self.handles[0]['item'])