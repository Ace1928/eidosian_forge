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
def inverseDataTransform(self):
    """Return the transform that maps from this image's local coordinate
        system to its input array.

        See dataTransform() for more information.

        :meta private:
        """
    return self._inverseDataTransform