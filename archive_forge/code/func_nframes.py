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
def nframes(self):
    """
        Returns
        -------
        int
            The number of frames in the image data.
        """
    if self.image is None:
        return 0
    elif self.axes['t'] is not None:
        return self.image.shape[self.axes['t']]
    return 1