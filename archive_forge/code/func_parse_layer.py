from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import assume
from hypothesis.strategies import tuples, integers, floats
from hypothesis.extra.numpy import arrays
def parse_layer(layer_data):
    x = layer_data[0, 1:]
    b = np.ascontiguousarray(layer_data[1:, 0], dtype='float64')
    W = layer_data[1:, 1:]
    assert x.ndim == 1
    assert b.ndim == 1
    assert b.shape[0] == W.shape[0]
    assert x.shape[0] == W.shape[1]
    assume(not np.isnan(W.sum()))
    assume(not np.isnan(x.sum()))
    assume(not np.isnan(b.sum()))
    assume(not any((np.isinf(val) for val in W.flatten())))
    assume(not any((np.isinf(val) for val in x)))
    assume(not any((np.isinf(val) for val in b)))
    return (x, b, W)