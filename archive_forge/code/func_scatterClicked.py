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
def scatterClicked(self, plt, points, ev):
    self.sigClicked.emit(self, ev)
    self.sigPointsClicked.emit(self, points, ev)