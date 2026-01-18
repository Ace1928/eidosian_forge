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
def downsampleMode(self):
    if self.ctrl.downsampleCheck.isChecked():
        ds = self.ctrl.downsampleSpin.value()
    else:
        ds = 1
    auto = self.ctrl.downsampleCheck.isChecked() and self.ctrl.autoDownsampleCheck.isChecked()
    if self.ctrl.subsampleRadio.isChecked():
        method = 'subsample'
    elif self.ctrl.meanRadio.isChecked():
        method = 'mean'
    elif self.ctrl.peakRadio.isChecked():
        method = 'peak'
    else:
        raise ValueError("one of the method radios must be selected for: 'subsample', 'mean', or 'peak'.")
    return (ds, auto, method)