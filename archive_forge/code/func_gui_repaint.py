import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
def gui_repaint(self, drawDC=None):
    """
        Update the displayed image on the GUI canvas, using the supplied
        wx.PaintDC device context.
        """
    _log.debug('%s - gui_repaint()', type(self))
    if not (self and self.IsShownOnScreen()):
        return
    if not drawDC:
        drawDC = wx.ClientDC(self)
    bmp = self.bitmap.ConvertToImage().ConvertToBitmap() if wx.Platform == '__WXMSW__' and isinstance(self.figure.canvas.get_renderer(), RendererWx) else self.bitmap
    drawDC.DrawBitmap(bmp, 0, 0)
    if self._rubberband_rect is not None:
        x0, y0, x1, y1 = map(round, self._rubberband_rect)
        rect = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x0, y0, x0, y1), (x0, y1, x1, y1)]
        drawDC.DrawLineList(rect, self._rubberband_pen_white)
        drawDC.DrawLineList(rect, self._rubberband_pen_black)