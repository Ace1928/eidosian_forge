from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import getConfigOption
from .. import parametertree as ptree
from ..graphicsItems.TextItem import TextItem
from ..Qt import QtCore, QtWidgets
from .ColorMapWidget import ColorMapParameter
from .DataFilterWidget import DataFilterParameter
from .PlotWidget import PlotWidget
def plotClicked(self, plot, points, ev):
    for pt in points:
        pt.originalIndex = self._visibleIndices[pt.index()]
    self.sigScatterPlotClicked.emit(self, points, ev)