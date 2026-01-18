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
def setDynamicRangeLimit(self, limit=1000000.0, hysteresis=3.0):
    """
        Limit the off-screen positions of data points at large magnification
        This avoids errors with plots not displaying because their visibility is incorrectly determined. 
        The default setting repositions far-off points to be within ±10^6 times the viewport height.

        =============== ================================================================
        **Arguments:**
        limit           (float or None) Any data outside the range of limit * hysteresis
                        will be constrained to the limit value limit.
                        All values are relative to the viewport height.
                        'None' disables the check for a minimal increase in performance.
                        Default is 1E+06.
                        
        hysteresis      (float) Hysteresis factor that controls how much change
                        in zoom level (vertical height) is allowed before recalculating
                        Default is 3.0
        =============== ================================================================
        """
    if hysteresis < 1.0:
        hysteresis = 1.0
    self.opts['dynamicRangeHyst'] = hysteresis
    if limit == self.opts['dynamicRangeLimit']:
        return
    self.opts['dynamicRangeLimit'] = limit
    self._datasetDisplay = None
    self.updateItems(styleUpdate=False)