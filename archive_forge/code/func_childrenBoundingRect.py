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
def childrenBoundingRect(self, *args, **kwds):
    range = self.childrenBounds(*args, **kwds)
    tr = self.targetRange()
    if range[0] is None:
        range[0] = tr[0]
    if range[1] is None:
        range[1] = tr[1]
    bounds = QtCore.QRectF(range[0][0], range[1][0], range[0][1] - range[0][0], range[1][1] - range[1][0])
    return bounds