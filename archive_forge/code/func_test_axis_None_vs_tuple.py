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
def test_axis_None_vs_tuple():
    shape = (3, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    res = stats.kruskal(*x, axis=None)
    res2 = stats.kruskal(*x, axis=(0, 1, 2))
    np.testing.assert_array_equal(res, res2)