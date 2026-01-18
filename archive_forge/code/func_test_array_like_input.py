from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
@pytest.mark.parametrize('dtype', list(np.typecodes['Float'] + np.typecodes['Integer'] + np.typecodes['Complex']))
def test_array_like_input(dtype):

    class ArrLike:

        def __init__(self, x):
            self._x = x

        def __array__(self):
            return np.asarray(x, dtype=dtype)
    x = [1] * 2 + [3, 4, 5]
    res = stats.mode(ArrLike(x))
    assert res.mode == 1
    assert res.count == 2