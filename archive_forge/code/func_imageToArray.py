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
def imageToArray(img, copy=False, transpose=True):
    """
    Convert a QImage into numpy array. The image must have format RGB32, ARGB32, or ARGB32_Premultiplied.
    By default, the image is not copied; changes made to the array will appear in the QImage as well (beware: if
    the QImage is collected before the array, there may be trouble).
    The array will have shape (width, height, (b,g,r,a)).
    """
    arr = ndarray_from_qimage(img)
    fmt = img.format()
    if fmt == img.Format.Format_RGB32:
        arr[..., 3] = 255
    if copy:
        arr = arr.copy()
    if transpose:
        return arr.transpose((1, 0, 2))
    else:
        return arr