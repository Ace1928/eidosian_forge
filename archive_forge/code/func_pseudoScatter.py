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
def pseudoScatter(data, spacing=None, shuffle=True, bidir=False, method='exact'):
    """Return an array of position values needed to make beeswarm or column scatter plots.
    
    Used for examining the distribution of values in an array.
    
    Given an array of x-values, construct an array of y-values such that an x,y scatter-plot
    will not have overlapping points (it will look similar to a histogram).
    """
    if method == 'exact':
        return _pseudoScatterExact(data, spacing=spacing, shuffle=shuffle, bidir=bidir)
    elif method == 'histogram':
        return _pseudoScatterHistogram(data, spacing=spacing, shuffle=shuffle, bidir=bidir)