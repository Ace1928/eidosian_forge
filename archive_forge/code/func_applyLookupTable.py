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
def applyLookupTable(data, lut):
    """
    Uses values in *data* as indexes to select values from *lut*.
    The returned data has shape data.shape + lut.shape[1:]
    
    Note: color gradient lookup tables can be generated using GradientWidget.

    Parameters
    ----------
    data : np.ndarray
    lut : np.ndarray
        Either cupy or numpy arrays are accepted, though this function has only
        consistently behaved correctly on windows with cuda toolkit version >= 11.1.
    """
    if data.dtype.kind not in ('i', 'u'):
        data = data.astype(int)
    cp = getCupy()
    if cp and cp.get_array_module(data) == cp:
        return cp.take(lut, cp.clip(data, 0, lut.shape[0] - 1), axis=0)
    else:
        return np.take(lut, data, axis=0, mode='clip')