import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
def test_KNeighborsClassifier_multioutput():
    rng = check_random_state(0)
    n_features = 5
    n_samples = 50
    n_output = 3
    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    weights = [None, 'uniform', 'distance', _weight_func]
    for algorithm, weights in product(ALGORITHMS, weights):
        y_pred_so = []
        y_pred_proba_so = []
        for o in range(n_output):
            knn = neighbors.KNeighborsClassifier(weights=weights, algorithm=algorithm)
            knn.fit(X_train, y_train[:, o])
            y_pred_so.append(knn.predict(X_test))
            y_pred_proba_so.append(knn.predict_proba(X_test))
        y_pred_so = np.vstack(y_pred_so).T
        assert y_pred_so.shape == y_test.shape
        assert len(y_pred_proba_so) == n_output
        knn_mo = neighbors.KNeighborsClassifier(weights=weights, algorithm=algorithm)
        knn_mo.fit(X_train, y_train)
        y_pred_mo = knn_mo.predict(X_test)
        assert y_pred_mo.shape == y_test.shape
        assert_array_equal(y_pred_mo, y_pred_so)
        y_pred_proba_mo = knn_mo.predict_proba(X_test)
        assert len(y_pred_proba_mo) == n_output
        for proba_mo, proba_so in zip(y_pred_proba_mo, y_pred_proba_so):
            assert_array_equal(proba_mo, proba_so)