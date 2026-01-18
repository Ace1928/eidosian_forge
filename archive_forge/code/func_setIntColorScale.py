from .. import functions as fn
from ..Qt import QtCore, QtGui
from .UIGraphicsItem import UIGraphicsItem
def setIntColorScale(self, minVal, maxVal, *args, **kargs):
    colors = [fn.intColor(i, maxVal - minVal, *args, **kargs) for i in range(minVal, maxVal)]
    g = QtGui.QLinearGradient()
    for i in range(len(colors)):
        x = float(i) / len(colors)
        g.setColorAt(x, colors[i])
    self.setGradient(g)
    if 'labels' not in kargs:
        self.setLabels({str(minVal): 0, str(maxVal): 1})
    else:
        self.setLabels({kargs['labels'][0]: 0, kargs['labels'][1]: 1})