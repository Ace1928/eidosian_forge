import weakref
import numpy as np
from .. import debug as debug
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .AxisItem import AxisItem
from .GradientEditorItem import GradientEditorItem
from .GraphicsWidget import GraphicsWidget
from .LinearRegionItem import LinearRegionItem
from .PlotCurveItem import PlotCurveItem
from .ViewBox import ViewBox
def imageChanged(self, autoLevel=False, autoRange=False):
    if self.imageItem() is None:
        return
    if self.levelMode == 'mono':
        for plt in self.plots[1:]:
            plt.setVisible(False)
        self.plots[0].setVisible(True)
        profiler = debug.Profiler()
        h = self.imageItem().getHistogram()
        profiler('get histogram')
        if h[0] is None:
            return
        self.plot.setData(*h)
        profiler('set plot')
        if autoLevel:
            mn = h[0][0]
            mx = h[0][-1]
            self.region.setRegion([mn, mx])
            profiler('set region')
        else:
            mn, mx = self.imageItem().getLevels()
            self.region.setRegion([mn, mx])
    else:
        self.plots[0].setVisible(False)
        ch = self.imageItem().getHistogram(perChannel=True)
        if ch[0] is None:
            return
        for i in range(1, 5):
            if len(ch) >= i:
                h = ch[i - 1]
                self.plots[i].setVisible(True)
                self.plots[i].setData(*h)
                if autoLevel:
                    mn = h[0][0]
                    mx = h[0][-1]
                    self.regions[i].setRegion([mn, mx])
            else:
                self.plots[i].setVisible(False)
        self._showRegions()