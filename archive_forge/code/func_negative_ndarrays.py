from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import assume
from hypothesis.strategies import tuples, integers, floats
from hypothesis.extra.numpy import arrays
def negative_ndarrays(min_len=0, max_len=10, min_val=-100000.0, dtype='float64'):
    return ndarrays(min_len=min_len, max_len=max_len, min_val=min_val, max_val=-1e-10, dtype=dtype)