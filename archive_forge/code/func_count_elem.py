import itertools
import numpy as np
import numpy.ma as ma
from numpy import ndarray, recarray
from numpy.ma import MaskedArray
from numpy.ma.mrecords import MaskedRecords
from numpy.core.overrides import array_function_dispatch
from numpy.lib._iotools import _is_string_like
def count_elem(dt):
    count = 1
    while dt.shape != ():
        for size in dt.shape:
            count *= size
        dt = dt.base
    return (dt, count)