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
def screenGeometry(self):
    """return the screen geometry of the viewbox"""
    v = self.getViewWidget()
    if v is None:
        return None
    b = self.sceneBoundingRect()
    wr = v.mapFromScene(b).boundingRect()
    pos = v.mapToGlobal(v.pos())
    wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
    return wr