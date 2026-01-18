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
def invertQTransform(tr):
    """Return a QTransform that is the inverse of *tr*.
    A pseudo-inverse is returned if tr is not invertible.
    
    Note that this function is preferred over QTransform.inverted() due to
    bugs in that method. (specifically, Qt has floating-point precision issues
    when determining whether a matrix is invertible)
    """
    try:
        det = tr.determinant()
        detr = 1.0 / det
        inv = tr.adjoint()
        inv *= detr
        return inv
    except ZeroDivisionError:
        return _pinv_fallback(tr)