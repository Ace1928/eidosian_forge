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
def affineSliceCoords(shape, origin, vectors, axes):
    """Return the array of coordinates used to sample data arrays in affineSlice().
    """
    if len(shape) != len(vectors):
        raise Exception('shape and vectors must have same length.')
    if len(origin) != len(axes):
        raise Exception('origin and axes must have same length.')
    for v in vectors:
        if len(v) != len(axes):
            raise Exception('each vector must be same length as axes.')
    shape = list(map(np.ceil, shape))
    if not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin)
    origin.shape = (len(axes),) + (1,) * len(shape)
    grid = np.mgrid[tuple([slice(0, x) for x in shape])]
    x = (grid[np.newaxis, ...] * vectors.transpose()[(Ellipsis,) + (np.newaxis,) * len(shape)]).sum(axis=1)
    x += origin
    return x