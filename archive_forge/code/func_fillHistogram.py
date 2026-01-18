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
def fillHistogram(self, fill=True, level=0.0, color=(100, 100, 200)):
    """Control fill of the histogram curve(s).

        Parameters
        ----------
        fill : bool, optional
            Set whether or not the histogram should be filled.
        level : float, optional
            Set the fill level. See :meth:`PlotCurveItem.setFillLevel
            <pyqtgraph.PlotCurveItem.setFillLevel>`. Only used if ``fill`` is True.
        color : color_like, optional
            Color to use for the fill when the histogram ``levelMode == "mono"``. See
            :meth:`PlotCurveItem.setBrush <pyqtgraph.PlotCurveItem.setBrush>`.
        """
    colors = [color, (255, 0, 0, 50), (0, 255, 0, 50), (0, 0, 255, 50), (255, 255, 255, 50)]
    for color, plot in zip(colors, self.plots):
        if fill:
            plot.setFillLevel(level)
            plot.setBrush(color)
        else:
            plot.setFillLevel(None)