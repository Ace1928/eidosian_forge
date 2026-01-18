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
def drawSymbol(painter, symbol, size, pen, brush):
    if symbol is None:
        return
    painter.scale(size, size)
    painter.setPen(pen)
    painter.setBrush(brush)
    if isinstance(symbol, str):
        symbol = Symbols[symbol]
    if np.isscalar(symbol):
        symbol = list(Symbols.values())[symbol % len(Symbols)]
    painter.drawPath(symbol)