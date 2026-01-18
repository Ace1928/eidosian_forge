import time
import numpy as np
import pyqtgraph as pg
from . import relax
def makeGraph(self):
    brushes = np.where(self.fixed, pg.mkBrush(0, 0, 0, 255), pg.mkBrush(50, 50, 200, 255))
    g2 = pg.GraphItem(pos=self.pos, adj=self.links[self.push & self.pull], pen=0.5, brush=brushes, symbol='o', size=self.mass ** 0.33, pxMode=False)
    p = pg.ItemGroup()
    p.addItem(g2)
    return p