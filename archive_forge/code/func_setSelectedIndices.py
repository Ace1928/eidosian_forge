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
def setSelectedIndices(self, inds):
    """Mark the specified indices as selected.

        Must be a sequence of integers that index into the array given in setData().
        """
    self.selectedIndices = inds
    self.updateSelected()