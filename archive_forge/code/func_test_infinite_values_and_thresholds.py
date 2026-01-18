import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.ensemble._hist_gradient_boosting._bitset import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
@pytest.mark.parametrize('num_threshold, expected_predictions', [(-np.inf, [0, 1, 1, 1]), (10, [0, 0, 1, 1]), (20, [0, 0, 0, 1]), (ALMOST_INF, [0, 0, 0, 1]), (np.inf, [0, 0, 0, 0])])
def test_infinite_values_and_thresholds(num_threshold, expected_predictions):
    X = np.array([-np.inf, 10, 20, np.inf]).reshape(-1, 1)
    nodes = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)
    nodes[0]['left'] = 1
    nodes[0]['right'] = 2
    nodes[0]['feature_idx'] = 0
    nodes[0]['num_threshold'] = num_threshold
    nodes[1]['is_leaf'] = True
    nodes[1]['value'] = 0
    nodes[2]['is_leaf'] = True
    nodes[2]['value'] = 1
    binned_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    raw_categorical_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    known_cat_bitset = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    f_idx_map = np.zeros(0, dtype=np.uint32)
    predictor = TreePredictor(nodes, binned_cat_bitsets, raw_categorical_bitsets)
    predictions = predictor.predict(X, known_cat_bitset, f_idx_map, n_threads)
    assert np.all(predictions == expected_predictions)