import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def scaleBy(self, s=None, center=None, x=None, y=None):
    """
        Scale by *s* around given center point (or center of view).
        *s* may be a Point or tuple (x, y).

        Optionally, x or y may be specified individually. This allows the other
        axis to be left unaffected (note that using a scale factor of 1.0 may
        cause slight changes due to floating-point error).
        """
    if s is not None:
        x, y = (s[0], s[1])
    affect = [x is not None, y is not None]
    if not any(affect):
        return
    scale = Point([1.0 if x is None else x, 1.0 if y is None else y])
    if self.state['aspectLocked'] is not False:
        scale[0] = scale[1]
    vr = self.targetRect()
    if center is None:
        center = Point(vr.center())
    else:
        center = Point(center)
    tl = center + (vr.topLeft() - center) * scale
    br = center + (vr.bottomRight() - center) * scale
    if not affect[0]:
        self.setYRange(tl.y(), br.y(), padding=0)
    elif not affect[1]:
        self.setXRange(tl.x(), br.x(), padding=0)
    else:
        self.setRange(QtCore.QRectF(tl, br), padding=0)