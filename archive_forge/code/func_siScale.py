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
def siScale(x, minVal=1e-25, allowUnicode=True):
    """
    Return the recommended scale factor and SI prefix string for x.
    
    Example::
    
        siScale(0.0001)   # returns (1e6, 'μ')
        # This indicates that the number 0.0001 is best represented as 0.0001 * 1e6 = 100 μUnits
    """
    if isinstance(x, decimal.Decimal):
        x = float(x)
    try:
        if not math.isfinite(x):
            return (1, '')
    except:
        raise
    if abs(x) < minVal:
        m = 0
    else:
        m = int(clip_scalar(math.floor(math.log(abs(x)) / math.log(1000)), -9.0, 9.0))
    if m == 0:
        pref = ''
    elif m < -8 or m > 8:
        pref = 'e%d' % (m * 3)
    elif allowUnicode:
        pref = SI_PREFIXES[m + 8]
    else:
        pref = SI_PREFIXES_ASCII[m + 8]
    m1 = -3 * m
    p = 10.0 ** m1
    return (p, pref)