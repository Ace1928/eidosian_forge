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
def setHistogramRange(self, mn, mx, padding=0.1):
    """Set the X/Y range on the histogram plot, depending on the orientation. This disables auto-scaling."""
    if self.orientation == 'vertical':
        self.vb.enableAutoRange(self.vb.YAxis, False)
        self.vb.setYRange(mn, mx, padding)
    else:
        self.vb.enableAutoRange(self.vb.XAxis, False)
        self.vb.setXRange(mn, mx, padding)