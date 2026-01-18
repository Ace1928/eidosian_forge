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
def getProcessedImage(self):
    """Returns the image data after it has been processed by any normalization options in use.
        """
    if self.imageDisp is None:
        image = self.normalize(self.image)
        self.imageDisp = image
        self._imageLevels = self.quickMinMax(self.imageDisp)
        self.levelMin = min([level[0] for level in self._imageLevels])
        self.levelMax = max([level[1] for level in self._imageLevels])
    return self.imageDisp