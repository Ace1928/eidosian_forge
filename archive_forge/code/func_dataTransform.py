import warnings
from collections.abc import Callable
import numpy
from .. import colormap
from .. import debug as debug
from .. import functions as fn
from .. import functions_qimage
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..util.cupy_helper import getCupy
from .GraphicsObject import GraphicsObject
def dataTransform(self):
    """
        Returns the transform that maps from this image's input array to its
        local coordinate system.

        This transform corrects for the transposition that occurs when image data
        is interpreted in row-major order.
        
        :meta private:
        """
    return self._dataTransform