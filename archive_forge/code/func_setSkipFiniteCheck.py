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
def setSkipFiniteCheck(self, skipFiniteCheck):
    """
        When it is known that the plot data passed to ``PlotDataItem`` contains only finite numerical values,
        the ``skipFiniteCheck`` property can help speed up plotting. If this flag is set and the data contains 
        any non-finite values (such as `NaN` or `Inf`), unpredictable behavior will occur. The data might not
        be plotted, or there migth be significant performance impact.
        
        In the default 'auto' connect mode, ``PlotDataItem`` will apply this setting automatically.
        """
    self.opts['skipFiniteCheck'] = bool(skipFiniteCheck)