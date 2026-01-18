import weakref
from math import ceil, floor, isfinite, log10, sqrt, frexp, floor
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget
def logTickValues(self, minVal, maxVal, size, stdTicks):
    ticks = []
    for spacing, t in stdTicks:
        if spacing >= 1.0:
            ticks.append((spacing, t))
    if len(ticks) < 3:
        v1 = int(floor(minVal))
        v2 = int(ceil(maxVal))
        minor = []
        for v in range(v1, v2):
            minor.extend(v + np.log10(np.arange(1, 10)))
        minor = [x for x in minor if x > minVal and x < maxVal]
        ticks.append((None, minor))
    return ticks