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
def mkColor(*args):
    """
    Convenience function for constructing QColor from a variety of argument 
    types. Accepted arguments are:
    
    ================ ================================================
     'c'             one of: r, g, b, c, m, y, k, w or an SVG color keyword
     R, G, B, [A]    integers 0-255
     (R, G, B, [A])  tuple of integers 0-255
     float           greyscale, 0.0-1.0
     int             see :func:`intColor() <pyqtgraph.intColor>`
     (int, hues)     see :func:`intColor() <pyqtgraph.intColor>`
     "#RGB"         
     "#RGBA"         
     "#RRGGBB"       
     "#RRGGBBAA"     
     QColor          QColor instance; makes a copy.
    ================ ================================================
    """
    err = 'Not sure how to make a color from "%s"' % str(args)
    if len(args) == 1:
        if isinstance(args[0], str):
            c = args[0]
            if len(c) == 1:
                try:
                    return QtGui.QColor(Colors[c])
                except KeyError:
                    raise ValueError('No color named "%s"' % c) from None
            if c[0] == '#' and len(c) < 10:
                c = c[1:]
                if len(c) < 6:
                    c = ''.join([x + x for x in c])
                return QtGui.QColor(*bytes.fromhex(c))
            else:
                qcol = QtGui.QColor(c)
                if qcol.isValid():
                    return qcol
                raise ValueError(f'Unable to convert {c} to QColor')
        elif isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        elif np.issubdtype(type(args[0]), np.floating):
            r = g = b = int(args[0] * 255)
            a = 255
        elif hasattr(args[0], '__len__'):
            if len(args[0]) == 3:
                r, g, b = args[0]
                a = 255
            elif len(args[0]) == 4:
                r, g, b, a = args[0]
            elif len(args[0]) == 2:
                return intColor(*args[0])
            else:
                raise TypeError(err)
        elif np.issubdtype(type(args[0]), np.integer):
            return intColor(args[0])
        else:
            raise TypeError(err)
    elif len(args) == 3:
        r, g, b = args
        a = 255
    elif len(args) == 4:
        r, g, b, a = args
    else:
        raise TypeError(err)
    args = [int(a) if np.isfinite(a) else 0 for a in (r, g, b, a)]
    return QtGui.QColor(*args)