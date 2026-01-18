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
def test_discrepancy(self):
    space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)
    space_2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]])
    space_2 = (2.0 * space_2 - 1.0) / (2.0 * 6.0)
    assert_allclose(qmc.discrepancy(space_1), 0.0081, atol=0.0001)
    assert_allclose(qmc.discrepancy(space_2), 0.0105, atol=0.0001)
    sample = np.array([[2, 1, 1, 2, 2, 2], [1, 2, 2, 2, 2, 2], [2, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 2], [1, 2, 2, 2, 1, 1], [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 2, 2]])
    sample = (2.0 * sample - 1.0) / (2.0 * 2.0)
    assert_allclose(qmc.discrepancy(sample, method='MD'), 2.5, atol=0.0001)
    assert_allclose(qmc.discrepancy(sample, method='WD'), 1.368, atol=0.0001)
    assert_allclose(qmc.discrepancy(sample, method='CD'), 0.3172, atol=0.0001)
    for dim in [2, 4, 8, 16, 32, 64]:
        ref = np.sqrt(3 ** (-dim))
        assert_allclose(qmc.discrepancy(np.array([[1] * dim]), method='L2-star'), ref)