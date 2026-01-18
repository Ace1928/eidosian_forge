import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
def processBypassed(self, args):
    if self.plot is None:
        return
    for item in list(self.items.values()):
        self.plot.removeItem(item)
    self.items = {}