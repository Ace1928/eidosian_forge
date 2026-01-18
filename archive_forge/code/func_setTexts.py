import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
def setTexts(self, text):
    for i in self.textItems:
        i.scene().removeItem(i)
    self.textItems = []
    for t in text:
        item = pg.TextItem(t)
        self.textItems.append(item)
        item.setParentItem(self)