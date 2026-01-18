import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1, min_pos
def test_min_pos():
    X = np.random.RandomState(0).randn(100)
    min_double = min_pos(X)
    min_float = min_pos(X.astype(np.float32))
    assert_allclose(min_double, min_float)
    assert min_double >= 0