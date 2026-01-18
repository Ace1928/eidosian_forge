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
def setFillLevel(self, level):
    """
        Enables filling the area under the curve towards the value specified by 
        `level`. `None` disables the filling. 
        """
    if self.opts['fillLevel'] == level:
        return
    self.opts['fillLevel'] = level
    self.updateItems(styleUpdate=True)