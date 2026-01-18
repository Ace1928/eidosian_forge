import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def setTickDensity(self, density=1.0):
    """
        The default behavior is to show at least two major ticks for axes of up to 300 pixels in length, 
        then add additional major ticks, spacing them out further as the available room increases.
        (Internally, the targeted number of major ticks grows with the square root of the axes length.)

        Setting a tick density different from the default value of `density = 1.0` scales the number of
        major ticks that is targeted for display. This only affects the automatic generation of ticks.
        """
    self._tickDensity = density
    self.picture = None
    self.update()