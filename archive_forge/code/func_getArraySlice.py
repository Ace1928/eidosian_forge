import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
def getArraySlice(self, data, img, axes=(0, 1), returnSlice=True):
    """Return a tuple of slice objects that can be used to slice the region
        from *data* that is covered by the bounding rectangle of this ROI.
        Also returns the transform that maps the ROI into data coordinates.
        
        If returnSlice is set to False, the function returns a pair of tuples with the values that would have 
        been used to generate the slice objects. ((ax0Start, ax0Stop), (ax1Start, ax1Stop))
        
        If the slice cannot be computed (usually because the scene/transforms are not properly
        constructed yet), then the method returns None.
        """
    dShape = (data.shape[axes[0]], data.shape[axes[1]])
    try:
        tr = self.sceneTransform() * fn.invertQTransform(img.sceneTransform())
    except np.linalg.linalg.LinAlgError:
        return None
    axisOrder = img.axisOrder
    if axisOrder == 'row-major':
        tr.scale(float(dShape[1]) / img.width(), float(dShape[0]) / img.height())
    else:
        tr.scale(float(dShape[0]) / img.width(), float(dShape[1]) / img.height())
    dataBounds = tr.mapRect(self.boundingRect())
    if axisOrder == 'row-major':
        intBounds = dataBounds.intersected(QtCore.QRectF(0, 0, dShape[1], dShape[0]))
    else:
        intBounds = dataBounds.intersected(QtCore.QRectF(0, 0, dShape[0], dShape[1]))
    bounds = ((int(min(intBounds.left(), intBounds.right())), int(1 + max(intBounds.left(), intBounds.right()))), (int(min(intBounds.bottom(), intBounds.top())), int(1 + max(intBounds.bottom(), intBounds.top()))))
    if axisOrder == 'row-major':
        bounds = bounds[::-1]
    if returnSlice:
        sl = [slice(None)] * data.ndim
        sl[axes[0]] = slice(*bounds[0])
        sl[axes[1]] = slice(*bounds[1])
        return (tuple(sl), tr)
    else:
        return (bounds, tr)