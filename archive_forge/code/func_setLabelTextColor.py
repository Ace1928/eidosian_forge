import math
from .. import functions as fn
from ..icons import invisibleEye
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .BarGraphItem import BarGraphItem
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .LabelItem import LabelItem
from .PlotDataItem import PlotDataItem
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
def setLabelTextColor(self, *args, **kargs):
    """Set the color of the item labels.

        Accepts the same arguments as :func:`~pyqtgraph.mkColor`.
        """
    self.opts['labelTextColor'] = fn.mkColor(*args, **kargs)
    for sample, label in self.items:
        label.setAttr('color', self.opts['labelTextColor'])
    self.update()