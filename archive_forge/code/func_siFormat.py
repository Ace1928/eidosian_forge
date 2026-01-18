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
def siFormat(x, precision=3, suffix='', space=True, error=None, minVal=1e-25, allowUnicode=True):
    """
    Return the number x formatted in engineering notation with SI prefix.
    
    Example::
        siFormat(0.0001, suffix='V')  # returns "100 μV"
    """
    if space is True:
        space = ' '
    if space is False:
        space = ''
    p, pref = siScale(x, minVal, allowUnicode)
    if not (len(pref) > 0 and pref[0] == 'e'):
        pref = space + pref
    if error is None:
        fmt = '%.' + str(precision) + 'g%s%s'
        return fmt % (x * p, pref, suffix)
    else:
        if allowUnicode:
            plusminus = space + '±' + space
        else:
            plusminus = ' +/- '
        fmt = '%.' + str(precision) + 'g%s%s%s%s'
        return fmt % (x * p, pref, suffix, plusminus, siFormat(error, precision=precision, suffix=suffix, space=space, minVal=minVal))