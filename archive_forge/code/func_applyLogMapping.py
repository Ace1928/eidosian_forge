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
def applyLogMapping(self, logMode):
    """
        Applies a logarithmic mapping transformation (base 10) if requested for the respective axis.
        This replaces the internal data. Values of ``-inf`` resulting from zeros in the original dataset are
        replaced by ``np.nan``.
        
        Parameters
        ----------
        logmode: tuple or list of two bool
            A `True` value requests log-scale mapping for the x and y axis (in this order).
        """
    if logMode[0]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.x = np.log10(self.x)
        nonfinites = ~np.isfinite(self.x)
        if nonfinites.any():
            self.x[nonfinites] = np.nan
            all_x_finite = False
        else:
            all_x_finite = True
        self.xAllFinite = all_x_finite
    if logMode[1]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.y = np.log10(self.y)
        nonfinites = ~np.isfinite(self.y)
        if nonfinites.any():
            self.y[nonfinites] = np.nan
            all_y_finite = False
        else:
            all_y_finite = True
        self.yAllFinite = all_y_finite