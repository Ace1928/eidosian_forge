import math
import sys
import weakref
from copy import deepcopy
import numpy as np
from ... import debug as debug
from ... import functions as fn
from ... import getConfigOption
from ...Point import Point
from ...Qt import QtCore, QtGui, QtWidgets, isQObjectAlive, QT_LIB
from ..GraphicsWidget import GraphicsWidget
from ..ItemGroup import ItemGroup
from .ViewBoxMenu import ViewBoxMenu
def suggestPadding(self, axis):
    l = self.width() if axis == 0 else self.height()
    def_pad = self.state['defaultPadding']
    if def_pad == 0.0:
        return def_pad
    max_pad = max(0.1, def_pad)
    if l > 0:
        padding = fn.clip_scalar(50 * def_pad / l ** 0.5, def_pad, max_pad)
    else:
        padding = def_pad
    return padding