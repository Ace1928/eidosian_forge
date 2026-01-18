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
def subArray(data, offset, shape, stride):
    """
    Unpack a sub-array from *data* using the specified offset, shape, and stride.
    
    Note that *stride* is specified in array elements, not bytes.
    For example, we have a 2x3 array packed in a 1D array as follows::
    
        data = [_, _, 00, 01, 02, _, 10, 11, 12, _]
        
    Then we can unpack the sub-array with this call::
    
        subArray(data, offset=2, shape=(2, 3), stride=(4, 1))
        
    ..which returns::
    
        [[00, 01, 02],
         [10, 11, 12]]
         
    This function operates only on the first axis of *data*. So changing 
    the input in the example above to have shape (10, 7) would cause the
    output to have shape (2, 3, 7).
    """
    data = np.ascontiguousarray(data)[offset:]
    shape = tuple(shape)
    extraShape = data.shape[1:]
    strides = list(data.strides[::-1])
    itemsize = strides[-1]
    for s in stride[1::-1]:
        strides.append(itemsize * s)
    strides = tuple(strides[::-1])
    return np.ndarray(buffer=data, shape=shape + extraShape, strides=strides, dtype=data.dtype)