import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def renderSymbol(symbol, size, pen, brush, device=None, dpr=1.0):
    """
    Render a symbol specification to QImage.
    Symbol may be either a QPainterPath or one of the keys in the Symbols dict.
    If *device* is None, a new QPixmap will be returned. Otherwise,
    the symbol will be rendered into the device specified (See QPainter documentation
    for more information).
    """
    penPxWidth = max(math.ceil(pen.widthF()), 1)
    if device is None:
        side = int(math.ceil(dpr * (size + penPxWidth)))
        device = QtGui.QImage(side, side, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        device.setDevicePixelRatio(dpr)
        device.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(device)
    try:
        p.setRenderHint(p.RenderHint.Antialiasing)
        p.translate(device.width() / dpr * 0.5, device.height() / dpr * 0.5)
        drawSymbol(p, symbol, size, pen, brush)
    finally:
        p.end()
    return device