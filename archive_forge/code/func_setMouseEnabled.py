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
def setMouseEnabled(self, x=None, y=None):
    """
        Set whether each axis is enabled for mouse interaction. *x*, *y* arguments must be True or False.
        This allows the user to pan/scale one axis of the view while leaving the other axis unchanged.
        """
    if x is not None:
        self.state['mouseEnabled'][0] = x
    if y is not None:
        self.state['mouseEnabled'][1] = y
    self.sigStateChanged.emit(self)