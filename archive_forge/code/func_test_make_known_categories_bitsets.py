import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.binning import (
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
def test_make_known_categories_bitsets():
    X = np.array([[14, 2, 30], [30, 4, 70], [40, 10, 180], [40, 240, 180]], dtype=X_DTYPE)
    bin_mapper = _BinMapper(n_bins=256, is_categorical=np.array([False, True, True]), known_categories=[None, X[:, 1], X[:, 2]])
    bin_mapper.fit(X)
    known_cat_bitsets, f_idx_map = bin_mapper.make_known_categories_bitsets()
    expected_f_idx_map = np.array([0, 0, 1], dtype=np.uint8)
    assert_allclose(expected_f_idx_map, f_idx_map)
    expected_cat_bitset = np.zeros((2, 8), dtype=np.uint32)
    f_idx = 1
    mapped_f_idx = f_idx_map[f_idx]
    expected_cat_bitset[mapped_f_idx, 0] = 2 ** 2 + 2 ** 4 + 2 ** 10
    expected_cat_bitset[mapped_f_idx, 7] = 2 ** 16
    f_idx = 2
    mapped_f_idx = f_idx_map[f_idx]
    expected_cat_bitset[mapped_f_idx, 0] = 2 ** 30
    expected_cat_bitset[mapped_f_idx, 2] = 2 ** 6
    expected_cat_bitset[mapped_f_idx, 5] = 2 ** 20
    assert_allclose(expected_cat_bitset, known_cat_bitsets)