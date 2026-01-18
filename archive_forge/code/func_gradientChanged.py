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
def gradientChanged(self):
    if self.imageItem() is not None:
        self._setImageLookupTable()
    self.lut = None
    self.sigLookupTableChanged.emit(self)