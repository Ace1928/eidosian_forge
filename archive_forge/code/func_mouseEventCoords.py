import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def mouseEventCoords(self, pos=None):
    """
        Calculate mouse coordinates in physical pixels.

        Qt uses logical pixels, but the figure is scaled to physical
        pixels for rendering.  Transform to physical pixels so that
        all of the down-stream transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
    if pos is None:
        pos = self.mapFromGlobal(QtGui.QCursor.pos())
    elif hasattr(pos, 'position'):
        pos = pos.position()
    elif hasattr(pos, 'pos'):
        pos = pos.pos()
    x = pos.x()
    y = self.figure.bbox.height / self.device_pixel_ratio - pos.y()
    return (x * self.device_pixel_ratio, y * self.device_pixel_ratio)