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
def test_lda_coefs():
    n_features = 2
    n_classes = 2
    n_samples = 1000
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=11)
    clf_lda_svd = LinearDiscriminantAnalysis(solver='svd')
    clf_lda_lsqr = LinearDiscriminantAnalysis(solver='lsqr')
    clf_lda_eigen = LinearDiscriminantAnalysis(solver='eigen')
    clf_lda_svd.fit(X, y)
    clf_lda_lsqr.fit(X, y)
    clf_lda_eigen.fit(X, y)
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_lsqr.coef_, 1)
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_eigen.coef_, 1)
    assert_array_almost_equal(clf_lda_eigen.coef_, clf_lda_lsqr.coef_, 1)