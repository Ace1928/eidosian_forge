import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
def test_integers_nd():
    d = 10
    rng = np.random.default_rng(3716505122102428560615700415287450951)
    low = rng.integers(low=-5, high=-1, size=d)
    high = rng.integers(low=1, high=5, size=d, endpoint=True)
    engine = RandomEngine(d, seed=rng)
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=False)
    assert_equal(sample.min(axis=0), low)
    assert_equal(sample.max(axis=0), high - 1)
    sample = engine.integers(low, u_bounds=high, n=100, endpoint=True)
    assert_equal(sample.min(axis=0), low)
    assert_equal(sample.max(axis=0), high)