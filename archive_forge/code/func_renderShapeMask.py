import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def renderShapeMask(self, width, height):
    """Return an array of 0.0-1.0 into which the shape of the item has been drawn.
        
        This can be used to mask array selections.
        """
    if width == 0 or height == 0:
        return np.empty((width, height), dtype=float)
    im = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
    im.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(im)
    p.setPen(fn.mkPen(None))
    p.setBrush(fn.mkBrush('w'))
    shape = self.shape()
    bounds = shape.boundingRect()
    p.scale(im.width() / bounds.width(), im.height() / bounds.height())
    p.translate(-bounds.topLeft())
    p.drawPath(shape)
    p.end()
    cidx = 0 if sys.byteorder == 'little' else 3
    mask = fn.ndarray_from_qimage(im)[..., cidx].T
    return mask.astype(float) / 255