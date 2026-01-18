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
def test_lloyd(self):
    rng = np.random.RandomState(1809831)
    sample = rng.uniform(0, 1, size=(128, 2))
    base_l1 = _l1_norm(sample)
    base_l2 = l2_norm(sample)
    for _ in range(4):
        sample_lloyd = _lloyd_centroidal_voronoi_tessellation(sample, maxiter=1)
        curr_l1 = _l1_norm(sample_lloyd)
        curr_l2 = l2_norm(sample_lloyd)
        assert base_l1 < curr_l1
        assert base_l2 < curr_l2
        base_l1 = curr_l1
        base_l2 = curr_l2
        sample = sample_lloyd