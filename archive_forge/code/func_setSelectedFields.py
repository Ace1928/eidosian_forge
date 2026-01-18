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
def setSelectedFields(self, *fields):
    self.fieldList.itemSelectionChanged.disconnect(self.fieldSelectionChanged)
    try:
        self.fieldList.clearSelection()
        for f in fields:
            i = list(self.fields.keys()).index(f)
            item = self.fieldList.item(i)
            item.setSelected(True)
    finally:
        self.fieldList.itemSelectionChanged.connect(self.fieldSelectionChanged)
    self.fieldSelectionChanged()