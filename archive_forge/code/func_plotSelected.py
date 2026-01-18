import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
def plotSelected(self, index):
    self.setPlot(self.ui.value())