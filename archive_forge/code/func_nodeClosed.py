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
def nodeClosed(self, node):
    del self._nodes[node.name()]
    self.widget().removeNode(node)
    for signal, slot in [('sigClosed', self.nodeClosed), ('sigRenamed', self.nodeRenamed), ('sigOutputChanged', self.nodeOutputChanged)]:
        try:
            getattr(node, signal).disconnect(slot)
        except (TypeError, RuntimeError):
            pass
    self.sigChartChanged.emit(self, 'remove', node)