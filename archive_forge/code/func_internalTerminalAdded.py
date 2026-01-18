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
def internalTerminalAdded(self, node, term):
    if term._io == 'in':
        io = 'out'
    else:
        io = 'in'
    Node.addTerminal(self, term.name(), io=io, renamable=term.isRenamable(), removable=term.isRemovable(), multiable=term.isMultiable())