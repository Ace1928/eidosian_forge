import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.utils._fast_dict import IntFloatDict, argmin
def test_int_float_dict_argmin():
    keys = np.arange(100, dtype=np.intp)
    values = np.arange(100, dtype=np.float64)
    d = IntFloatDict(keys, values)
    assert argmin(d) == (0, 0)