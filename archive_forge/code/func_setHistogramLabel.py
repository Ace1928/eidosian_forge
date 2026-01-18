import os
from math import log10
from time import perf_counter
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..graphicsItems.GradientEditorItem import addGradientListToDocstring
from ..graphicsItems.ImageItem import ImageItem
from ..graphicsItems.InfiniteLine import InfiniteLine
from ..graphicsItems.LinearRegionItem import LinearRegionItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..graphicsItems.VTickGroup import VTickGroup
from ..Qt import QtCore, QtGui, QtWidgets
from ..SignalProxy import SignalProxy
from . import ImageViewTemplate_generic as ui_template
def setHistogramLabel(self, text=None, **kwargs):
    """
        Set the label text of the histogram axis similar to
        :func:`AxisItem.setLabel() <pyqtgraph.AxisItem.setLabel>`
        """
    a = self.ui.histogram.axis
    a.setLabel(text, **kwargs)
    if text == '':
        a.showLabel(False)
    self.ui.histogram.setMinimumWidth(135)