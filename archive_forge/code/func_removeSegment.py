import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def removeSegment(self, seg):
    for handle in seg.handles[:]:
        seg.removeHandle(handle['item'])
    self.segments.remove(seg)
    seg.sigClicked.disconnect(self.segmentClicked)
    self.scene().removeItem(seg)