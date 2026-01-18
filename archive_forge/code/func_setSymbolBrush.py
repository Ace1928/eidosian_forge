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
def setSymbolBrush(self, *args, **kargs):
    """
        Sets the :class:`QtGui.QBrush` used to fill symbols.
        See :func:`mkBrush() <pyqtgraph.mkBrush>`) for arguments.
        """
    brush = fn.mkBrush(*args, **kargs)
    if self.opts['symbolBrush'] == brush:
        return
    self.opts['symbolBrush'] = brush
    self.updateItems(styleUpdate=True)