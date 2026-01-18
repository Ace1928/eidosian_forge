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
def setCurveClickable(self, state, width=None):
    """ ``state=True`` sets the curve to be clickable, with a tolerance margin represented by `width`. """
    self.curve.setClickable(state, width)