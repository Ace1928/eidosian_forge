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
def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
    """
        Enable (or disable) auto-range for *axis*, which may be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes for both
        (if *axis* is omitted, both axes will be changed).
        When enabled, the axis will automatically rescale when items are added/removed or change their shape.
        The argument *enable* may optionally be a float (0.0-1.0) which indicates the fraction of the data that should
        be visible (this only works with items implementing a dataBounds method, such as PlotDataItem).
        """
    if x is not None or y is not None:
        if x is not None:
            self.enableAutoRange(ViewBox.XAxis, x)
        if y is not None:
            self.enableAutoRange(ViewBox.YAxis, y)
        return
    if enable is True:
        enable = 1.0
    if axis is None:
        axis = ViewBox.XYAxes
    if axis == ViewBox.XYAxes or axis == 'xy':
        axes = [0, 1]
    elif axis == ViewBox.XAxis or axis == 'x':
        axes = [0]
    elif axis == ViewBox.YAxis or axis == 'y':
        axes = [1]
    else:
        raise Exception('axis argument must be ViewBox.XAxis, ViewBox.YAxis, or ViewBox.XYAxes.')
    for ax in axes:
        if self.state['autoRange'][ax] != enable:
            if enable is False and self._autoRangeNeedsUpdate:
                self.updateAutoRange()
            self.state['autoRange'][ax] = enable
            self._autoRangeNeedsUpdate |= enable is not False
            self.update()
    self.sigStateChanged.emit(self)