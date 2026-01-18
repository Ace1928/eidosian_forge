import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def setWidth(self, w=None):
    """Set the width of this axis reserved for ticks and tick labels.
        The width of the axis label is automatically added.

        If *width* is None, then the value will be determined automatically
        based on the size of the tick text."""
    self.fixedWidth = w
    self._updateWidth()