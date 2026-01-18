import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
def removeTick(self, tick, finish=True):
    """
        Removes the specified tick.
        """
    del self.ticks[tick]
    tick.setParentItem(None)
    if self.scene() is not None:
        self.scene().removeItem(tick)
    self.sigTicksChanged.emit(self)
    if finish:
        self.sigTicksChangeFinished.emit(self)