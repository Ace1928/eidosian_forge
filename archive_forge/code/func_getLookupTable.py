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
def getLookupTable(self, img=None, n=None, alpha=None):
    """Return a lookup table from the color gradient defined by this
        HistogramLUTItem.
        """
    if self.levelMode != 'mono':
        return None
    if n is None:
        if img.dtype == np.uint8:
            n = 256
        else:
            n = 512
    if self.lut is None:
        self.lut = self.gradient.getLookupTable(n, alpha=alpha)
    return self.lut