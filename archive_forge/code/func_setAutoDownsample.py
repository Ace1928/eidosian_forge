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
def setAutoDownsample(self, active=True):
    """
        Controls automatic downsampling for this ImageItem.

        If `active` is `True`, the image is automatically downsampled to match the
        screen resolution. This improves performance for large images and
        reduces aliasing. If `autoDownsample` is not specified, then ImageItem will
        choose whether to downsample the image based on its size.
        
        `False` disables automatic downsampling.
        """
    self.autoDownsample = active
    self._renderRequired = True
    self.update()