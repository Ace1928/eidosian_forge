import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def replace_masked(s):
    if np.ma.is_masked(s):
        rep = ~np.all(asorted.mask, axis=axis, keepdims=True) & s.mask
        s.data[rep] = np.ma.minimum_fill_value(asorted)
        s.mask[rep] = False