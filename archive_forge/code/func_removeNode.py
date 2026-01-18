import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
def removeNode(self, node):
    if node in self.items:
        item = self.items[node]
        try:
            item.bypassBtn.clicked.disconnect(self.bypassClicked)
        except (TypeError, RuntimeError):
            pass
        self.ui.ctrlList.removeTopLevelItem(item)