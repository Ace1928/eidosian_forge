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
def test_discrepancy_errors(self):
    sample = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    with pytest.raises(ValueError, match='Sample is not in unit hypercube'):
        qmc.discrepancy(sample)
    with pytest.raises(ValueError, match='Sample is not a 2D array'):
        qmc.discrepancy([1, 3])
    sample = [[0, 0], [1, 1], [0.5, 0.5]]
    with pytest.raises(ValueError, match="'toto' is not a valid ..."):
        qmc.discrepancy(sample, method='toto')