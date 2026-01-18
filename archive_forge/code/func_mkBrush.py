from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def mkBrush(*args, **kwds):
    """
    | Convenience function for constructing Brush.
    | This function always constructs a solid brush and accepts the same arguments as :func:`mkColor() <pyqtgraph.mkColor>`
    | Calling mkBrush(None) returns an invisible brush.
    """
    if 'color' in kwds:
        color = kwds['color']
    elif len(args) == 1:
        arg = args[0]
        if arg is None:
            return QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush)
        elif isinstance(arg, QtGui.QBrush):
            return QtGui.QBrush(arg)
        else:
            color = arg
    elif len(args) > 1:
        color = args
    return QtGui.QBrush(mkColor(color))