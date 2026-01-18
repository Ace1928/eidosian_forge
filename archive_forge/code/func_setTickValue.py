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
def setTickValue(self, tick, val):
    """
        Set the position (along the slider) of the tick.
        
        ==============   ==================================================================
        **Arguments:**
        tick             Can be either an integer corresponding to the index of the tick
                         or a Tick object. Ex: if you had a slider with 3 ticks and you
                         wanted to change the middle tick, the index would be 1.
        val              The desired position of the tick. If val is < 0, position will be
                         set to 0. If val is > 1, position will be set to 1.
        ==============   ==================================================================
        """
    tick = self.getTick(tick)
    val = min(max(0.0, val), 1.0)
    x = val * self.length
    pos = tick.pos()
    pos.setX(x)
    tick.setPos(pos)
    self.ticks[tick] = val
    self.update()
    self.sigTicksChanged.emit(self)
    self.sigTicksChangeFinished.emit(self)