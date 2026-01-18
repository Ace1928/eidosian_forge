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
def fieldSelectionChanged(self):
    sel = self.fieldList.selectedItems()
    if len(sel) > 2:
        self.fieldList.blockSignals(True)
        try:
            for item in sel[1:-1]:
                item.setSelected(False)
        finally:
            self.fieldList.blockSignals(False)
    self.updatePlot()