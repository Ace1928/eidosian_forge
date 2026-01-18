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
def setShadowPen(self, *args, **kargs):
    """
        Sets the shadow pen used to draw lines between points (this is for enhancing contrast or
        emphasizing data). This line is drawn behind the primary pen and should generally be assigned 
        greater width than the primary pen.
        The argument can be a :class:`QtGui.QPen` or any combination of arguments accepted by 
        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`.
        """
    if args and args[0] is None:
        pen = None
    else:
        pen = fn.mkPen(*args, **kargs)
    self.opts['shadowPen'] = pen
    self.updateItems(styleUpdate=True)