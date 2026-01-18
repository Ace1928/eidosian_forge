import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def setCode(self, code):
    ind = []
    lines = code.split('\n')
    for line in lines:
        stripped = line.lstrip()
        if len(stripped) > 0:
            ind.append(len(line) - len(stripped))
    if len(ind) > 0:
        ind = min(ind)
        code = '\n'.join([line[ind:] for line in lines])
    self.text.clear()
    self.text.insertPlainText(code)