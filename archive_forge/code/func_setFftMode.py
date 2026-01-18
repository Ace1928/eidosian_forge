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
def setFftMode(self, state):
    """
        ``state = True`` enables mapping the data by a fast Fourier transform.
        If the `x` values are not equidistant, the data set is resampled at
        equal intervals. 
        """
    if self.opts['fftMode'] == state:
        return
    self.opts['fftMode'] = state
    self._datasetMapped = None
    self._datasetDisplay = None
    self.updateItems(styleUpdate=False)
    self.informViewBoundsChanged()