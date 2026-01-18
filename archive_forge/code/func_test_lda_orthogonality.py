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
def test_lda_orthogonality():
    means = np.array([[0, 0, -1], [0, 2, 0], [0, -2, 0], [0, 0, 5]])
    scatter = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.1], [0, 0, -0.1]])
    X = (means[:, np.newaxis, :] + scatter[np.newaxis, :, :]).reshape((-1, 3))
    y = np.repeat(np.arange(means.shape[0]), scatter.shape[0])
    clf = LinearDiscriminantAnalysis(solver='svd').fit(X, y)
    means_transformed = clf.transform(means)
    d1 = means_transformed[3] - means_transformed[0]
    d2 = means_transformed[2] - means_transformed[1]
    d1 /= np.sqrt(np.sum(d1 ** 2))
    d2 /= np.sqrt(np.sum(d2 ** 2))
    assert_almost_equal(np.cov(clf.transform(scatter).T), np.eye(2))
    assert_almost_equal(np.abs(np.dot(d1[:2], [1, 0])), 1.0)
    assert_almost_equal(np.abs(np.dot(d2[:2], [0, 1])), 1.0)