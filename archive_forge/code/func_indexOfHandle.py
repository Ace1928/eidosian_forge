import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def indexOfHandle(self, handle):
    """
        Return the index of *handle* in the list of this ROI's handles.
        """
    if isinstance(handle, Handle):
        index = [i for i, info in enumerate(self.handles) if info['item'] is handle]
        if len(index) == 0:
            raise Exception('Cannot return handle index; not attached to this ROI')
        return index[0]
    else:
        return handle