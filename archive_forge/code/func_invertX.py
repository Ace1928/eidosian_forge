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
def invertX(self, b=True):
    """
        By default, the positive x-axis points rightward on the screen. Use invertX(True) to reverse the x-axis.
        """
    self._invertAxis(0, b)