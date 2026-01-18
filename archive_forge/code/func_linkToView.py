import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def linkToView(self, view):
    """Link this axis to a ViewBox, causing its displayed range to match the visible range of the view."""
    self._linkToView_internal(view)