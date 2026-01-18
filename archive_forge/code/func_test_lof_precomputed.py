import re
from math import sqrt
import numpy as np
import pytest
from sklearn import metrics, neighbors
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_lof_precomputed(global_dtype, random_state=42):
    """Tests LOF with a distance matrix."""
    rng = np.random.RandomState(random_state)
    X = rng.random_sample((10, 4)).astype(global_dtype, copy=False)
    Y = rng.random_sample((3, 4)).astype(global_dtype, copy=False)
    DXX = metrics.pairwise_distances(X, metric='euclidean')
    DYX = metrics.pairwise_distances(Y, X, metric='euclidean')
    lof_X = neighbors.LocalOutlierFactor(n_neighbors=3, novelty=True)
    lof_X.fit(X)
    pred_X_X = lof_X._predict()
    pred_X_Y = lof_X.predict(Y)
    lof_D = neighbors.LocalOutlierFactor(n_neighbors=3, algorithm='brute', metric='precomputed', novelty=True)
    lof_D.fit(DXX)
    pred_D_X = lof_D._predict()
    pred_D_Y = lof_D.predict(DYX)
    assert_allclose(pred_X_X, pred_D_X)
    assert_allclose(pred_X_Y, pred_D_Y)