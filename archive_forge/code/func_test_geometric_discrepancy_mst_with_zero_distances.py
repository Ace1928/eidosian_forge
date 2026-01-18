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
@pytest.mark.xfail(reason='minimum_spanning_tree ignores zero distances (#18892)', strict=True)
def test_geometric_discrepancy_mst_with_zero_distances(self):
    sample = np.array([[0, 0], [0, 0], [0, 1]])
    assert_allclose(qmc.geometric_discrepancy(sample, method='mst'), 0.5)