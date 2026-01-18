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
def drawRectangle(self, rect):
    if rect is not None:
        x0, y0, w, h = [int(pt / self.device_pixel_ratio) for pt in rect]
        x1 = x0 + w
        y1 = y0 + h

        def _draw_rect_callback(painter):
            pen = QtGui.QPen(QtGui.QColor('black'), 1 / self.device_pixel_ratio)
            pen.setDashPattern([3, 3])
            for color, offset in [(QtGui.QColor('black'), 0), (QtGui.QColor('white'), 3)]:
                pen.setDashOffset(offset)
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawLine(x0, y0, x0, y1)
                painter.drawLine(x0, y0, x1, y0)
                painter.drawLine(x0, y1, x1, y1)
                painter.drawLine(x1, y0, x1, y1)
    else:

        def _draw_rect_callback(painter):
            return
    self._draw_rect_callback = _draw_rect_callback
    self.update()