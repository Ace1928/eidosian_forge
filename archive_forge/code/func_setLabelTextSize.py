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
def setLabelTextSize(self, size):
    """Set the `size` of the item labels.

        Accepts the CSS style string arguments, e.g. '8pt'.
        """
    self.opts['labelTextSize'] = size
    for _, label in self.items:
        label.setAttr('size', self.opts['labelTextSize'])
    self.update()