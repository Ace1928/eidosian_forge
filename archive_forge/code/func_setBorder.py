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
def setBorder(self, *args, **kwds):
    """
        Set the pen used to draw border around the view

        If border is None, then no border will be drawn.

        Added in version 0.9.10

        See :func:`mkPen <pyqtgraph.mkPen>` for arguments.
        """
    self.border = fn.mkPen(*args, **kwds)
    self.borderRect.setPen(self.border)