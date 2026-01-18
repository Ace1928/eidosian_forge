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
def test_lda_ledoitwolf():

    class StandardizedLedoitWolf:

        def fit(self, X):
            sc = StandardScaler()
            X_sc = sc.fit_transform(X)
            s = ledoit_wolf(X_sc)[0]
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            self.covariance_ = s
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100,))
    c1 = LinearDiscriminantAnalysis(store_covariance=True, shrinkage='auto', solver='lsqr')
    c2 = LinearDiscriminantAnalysis(store_covariance=True, covariance_estimator=StandardizedLedoitWolf(), solver='lsqr')
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)