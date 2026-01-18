import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def terminalRenamed(self, term, oldName):
    Node.terminalRenamed(self, term, oldName)
    item = term.joinItem
    item.setText(0, term.name())
    self.update()