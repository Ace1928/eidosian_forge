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
def setMenuEnabled(self, enableMenu=True, enableViewBoxMenu='same'):
    """
        Enable or disable the context menu for this PlotItem.
        By default, the ViewBox's context menu will also be affected.
        (use ``enableViewBoxMenu=None`` to leave the ViewBox unchanged)
        """
    self._menuEnabled = enableMenu
    if enableViewBoxMenu is None:
        return
    if enableViewBoxMenu == 'same':
        enableViewBoxMenu = enableMenu
    self.vb.setMenuEnabled(enableViewBoxMenu)