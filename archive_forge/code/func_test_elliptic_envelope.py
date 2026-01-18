import numpy as np
import pytest
from sklearn.covariance import EllipticEnvelope
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
def test_elliptic_envelope(global_random_seed):
    rnd = np.random.RandomState(global_random_seed)
    X = rnd.randn(100, 10)
    clf = EllipticEnvelope(contamination=0.1)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.raises(NotFittedError):
        clf.decision_function(X)
    clf.fit(X)
    y_pred = clf.predict(X)
    scores = clf.score_samples(X)
    decisions = clf.decision_function(X)
    assert_array_almost_equal(scores, -clf.mahalanobis(X))
    assert_array_almost_equal(clf.mahalanobis(X), clf.dist_)
    assert_almost_equal(clf.score(X, np.ones(100)), (100 - y_pred[y_pred == -1].size) / 100.0)
    assert sum(y_pred == -1) == sum(decisions < 0)