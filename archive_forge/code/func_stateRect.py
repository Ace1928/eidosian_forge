import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def stateRect(self, state):
    r = QtCore.QRectF(0, 0, state['size'][0], state['size'][1])
    tr = QtGui.QTransform()
    tr.rotate(-state['angle'])
    r = tr.mapRect(r)
    return r.adjusted(state['pos'][0], state['pos'][1], state['pos'][0], state['pos'][1])