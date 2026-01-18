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
def quickMinMax(self, data):
    """
        Estimate the min/max values of *data* by subsampling.
        Returns [(min, max), ...] with one item per channel
        """
    while data.size > 1000000.0:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[tuple(sl)]
    cax = self.axes['c']
    if cax is None:
        if data.size == 0:
            return [(0, 0)]
        return [(float(nanmin(data)), float(nanmax(data)))]
    else:
        if data.size == 0:
            return [(0, 0)] * data.shape[-1]
        return [(float(nanmin(data.take(i, axis=cax))), float(nanmax(data.take(i, axis=cax)))) for i in range(data.shape[-1])]