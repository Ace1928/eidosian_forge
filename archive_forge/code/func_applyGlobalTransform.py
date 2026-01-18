import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def applyGlobalTransform(self, tr):
    st = self.getState()
    st['scale'] = st['size']
    st = SRTTransform(st)
    st = (st * tr).saveState()
    st['size'] = st['scale']
    self.setState(st)