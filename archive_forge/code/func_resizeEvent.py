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
def resizeEvent(self, event):
    if self._in_resize_event:
        return
    self._in_resize_event = True
    try:
        w = event.size().width() * self.device_pixel_ratio
        h = event.size().height() * self.device_pixel_ratio
        dpival = self.figure.dpi
        winch = w / dpival
        hinch = h / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        QtWidgets.QWidget.resizeEvent(self, event)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()
    finally:
        self._in_resize_event = False