import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
def test_lda_predict():
    for test_case in solver_shrinkage:
        solver, shrinkage = test_case
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        y_pred = clf.fit(X, y).predict(X)
        assert_array_equal(y_pred, y, 'solver %s' % solver)
        y_pred1 = clf.fit(X1, y).predict(X1)
        assert_array_equal(y_pred1, y, 'solver %s' % solver)
        y_proba_pred1 = clf.predict_proba(X1)
        assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y, 'solver %s' % solver)
        y_log_proba_pred1 = clf.predict_log_proba(X1)
        assert_allclose(np.exp(y_log_proba_pred1), y_proba_pred1, rtol=1e-06, atol=1e-06, err_msg='solver %s' % solver)
        y_pred3 = clf.fit(X, y3).predict(X)
        assert np.any(y_pred3 != y3), 'solver %s' % solver
    clf = LinearDiscriminantAnalysis(solver='svd', shrinkage='auto')
    with pytest.raises(NotImplementedError):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.1, covariance_estimator=ShrunkCovariance())
    with pytest.raises(ValueError, match='covariance_estimator and shrinkage parameters are not None. Only one of the two can be set.'):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(solver='svd', covariance_estimator=LedoitWolf())
    with pytest.raises(ValueError, match='covariance estimator is not supported with svd'):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=KMeans(n_clusters=2, n_init='auto'))
    with pytest.raises(ValueError):
        clf.fit(X, y)