import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def checkRemoveHandle(self, h):
    if self.closed:
        return len(self.handles) > 3
    else:
        return len(self.handles) > 2