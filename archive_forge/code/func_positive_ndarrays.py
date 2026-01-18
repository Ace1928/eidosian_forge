from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import assume
from hypothesis.strategies import tuples, integers, floats
from hypothesis.extra.numpy import arrays
def positive_ndarrays(min_len=0, max_len=10, max_val=100000.0, dtype='float64'):
    return ndarrays(min_len=min_len, max_len=max_len, min_val=0, max_val=max_val, dtype=dtype)