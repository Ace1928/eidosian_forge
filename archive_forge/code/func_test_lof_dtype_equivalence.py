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
@pytest.mark.parametrize('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
@pytest.mark.parametrize('novelty', [True, False])
@pytest.mark.parametrize('contamination', [0.5, 'auto'])
def test_lof_dtype_equivalence(algorithm, novelty, contamination):
    """Check the equivalence of the results with 32 and 64 bits input."""
    inliers = iris.data[:50]
    outliers = iris.data[-5:]
    X = np.concatenate([inliers, outliers], axis=0).astype(np.float32)
    lof_32 = neighbors.LocalOutlierFactor(algorithm=algorithm, novelty=novelty, contamination=contamination)
    X_32 = X.astype(np.float32, copy=True)
    lof_32.fit(X_32)
    lof_64 = neighbors.LocalOutlierFactor(algorithm=algorithm, novelty=novelty, contamination=contamination)
    X_64 = X.astype(np.float64, copy=True)
    lof_64.fit(X_64)
    assert_allclose(lof_32.negative_outlier_factor_, lof_64.negative_outlier_factor_)
    for method in ('score_samples', 'decision_function', 'predict', 'fit_predict'):
        if hasattr(lof_32, method):
            y_pred_32 = getattr(lof_32, method)(X_32)
            y_pred_64 = getattr(lof_64, method)(X_64)
            assert_allclose(y_pred_32, y_pred_64, atol=0.0002)