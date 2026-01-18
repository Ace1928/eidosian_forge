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
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_MultinomialBasicDraw(self):
    seed = np.random.default_rng(6955663962957011631562466584467607969)
    p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
    n_trials = 100
    expected = np.atleast_2d(n_trials * p).astype(int)
    engine = qmc.MultinomialQMC(p, n_trials=n_trials, seed=seed)
    assert_allclose(engine.random(1), expected, atol=1)