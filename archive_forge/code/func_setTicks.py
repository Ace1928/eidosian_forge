import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
def setTicks(self, ticks):
    for tick, pos in self.listTicks():
        self.removeTick(tick)
    for pos in ticks:
        tickItem = self.addTick(pos, movable=False, color='#333333')
        self.all_ticks[pos] = tickItem
    self.updateRange(None, self._range)