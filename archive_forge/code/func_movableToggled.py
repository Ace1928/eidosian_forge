import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def movableToggled(self, b):
    for item in self.items.values():
        item.setMovable(b)