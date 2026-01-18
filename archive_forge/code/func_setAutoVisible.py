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
def setAutoVisible(self, x=None, y=None):
    """Set whether automatic range uses only visible data when determining
        the range to show.
        """
    if x is not None:
        self.state['autoVisibleOnly'][0] = x
        if x is True:
            self.state['autoVisibleOnly'][1] = False
    if y is not None:
        self.state['autoVisibleOnly'][1] = y
        if y is True:
            self.state['autoVisibleOnly'][0] = False
    if x is not None or y is not None:
        self.updateAutoRange()