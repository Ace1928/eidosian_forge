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
def interweaveArrays(*args):
    """
    Parameters
    ----------

    args : numpy.ndarray
           series of 1D numpy arrays of the same length and dtype
    
    Returns
    -------
    numpy.ndarray
        A numpy array with all the input numpy arrays interwoven

    Examples
    --------

    >>> result = interweaveArrays(numpy.ndarray([0, 2, 4]), numpy.ndarray([1, 3, 5]))
    >>> result
    array([0, 1, 2, 3, 4, 5])
    """
    size = sum((x.size for x in args))
    result = np.empty((size,), dtype=args[0].dtype)
    n = len(args)
    for index, array in enumerate(args):
        result[index::n] = array
    return result