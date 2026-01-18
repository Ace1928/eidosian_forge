import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
def setSymbolPen(self, *args, **kargs):
    """ 
        Sets the :class:`QtGui.QPen` used to draw symbol outlines.
        See :func:`mkPen() <pyqtgraph.mkPen>`) for arguments.
        """
    pen = fn.mkPen(*args, **kargs)
    if self.opts['symbolPen'] == pen:
        return
    self.opts['symbolPen'] = pen
    self.updateItems(styleUpdate=True)