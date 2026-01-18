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
def clip_array(arr, vmin, vmax, out=None):
    if vmin is None and vmax is None:
        return np.clip(arr, vmin, vmax, out=out)
    if vmin is None:
        return np.core.umath.minimum(arr, vmax, out=out)
    elif vmax is None:
        return np.core.umath.maximum(arr, vmin, out=out)
    else:
        return np.core.umath.clip(arr, vmin, vmax, out=out)