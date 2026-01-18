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
def test_lof_input_dtype_preservation(global_dtype, algorithm, contamination, novelty):
    """Check that the fitted attributes are stored using the data type of X."""
    X = iris.data.astype(global_dtype, copy=False)
    iso = neighbors.LocalOutlierFactor(n_neighbors=5, algorithm=algorithm, contamination=contamination, novelty=novelty)
    iso.fit(X)
    assert iso.negative_outlier_factor_.dtype == global_dtype
    for method in ('score_samples', 'decision_function'):
        if hasattr(iso, method):
            y_pred = getattr(iso, method)(X)
            assert y_pred.dtype == global_dtype