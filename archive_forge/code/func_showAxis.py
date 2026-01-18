import collections.abc
import os
import warnings
import weakref
import numpy as np
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox
from . import plotConfigTemplate_generic as ui_template
def showAxis(self, axis, show=True):
    """
        Show or hide one of the plot's axes.
        axis must be one of 'left', 'bottom', 'right', or 'top'
        """
    s = self.getScale(axis)
    if show:
        s.show()
    else:
        s.hide()