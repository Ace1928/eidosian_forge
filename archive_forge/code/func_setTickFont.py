import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def setTickFont(self, font):
    """
        (QFont or None) Determines the font used for tick values. 
        Use None for the default font.
        """
    self.style['tickFont'] = font
    self.picture = None
    self.prepareGeometryChange()
    self.update()