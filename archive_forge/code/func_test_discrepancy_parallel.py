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
def test_discrepancy_parallel(self, monkeypatch):
    sample = np.array([[2, 1, 1, 2, 2, 2], [1, 2, 2, 2, 2, 2], [2, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 2], [1, 2, 2, 2, 1, 1], [2, 2, 2, 2, 1, 1], [2, 2, 2, 1, 2, 2]])
    sample = (2.0 * sample - 1.0) / (2.0 * 2.0)
    assert_allclose(qmc.discrepancy(sample, method='MD', workers=8), 2.5, atol=0.0001)
    assert_allclose(qmc.discrepancy(sample, method='WD', workers=8), 1.368, atol=0.0001)
    assert_allclose(qmc.discrepancy(sample, method='CD', workers=8), 0.3172, atol=0.0001)
    for dim in [2, 4, 8, 16, 32, 64]:
        ref = np.sqrt(3 ** (-dim))
        assert_allclose(qmc.discrepancy(np.array([[1] * dim]), method='L2-star', workers=-1), ref)
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    with pytest.raises(NotImplementedError, match='Cannot determine the'):
        qmc.discrepancy(sample, workers=-1)
    with pytest.raises(ValueError, match='Invalid number of workers...'):
        qmc.discrepancy(sample, workers=-2)