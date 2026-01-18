import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def itemChanged(self, item):
    col = str(item.text())
    if item.checkState() == QtCore.Qt.CheckState.Checked:
        if col not in self.columns:
            self.columns.add(col)
            self.addOutput(col)
    elif col in self.columns:
        self.columns.remove(col)
        self.removeTerminal(col)
    self.update()