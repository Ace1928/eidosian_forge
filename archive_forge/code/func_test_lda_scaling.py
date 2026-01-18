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
def test_lda_scaling():
    n = 100
    rng = np.random.RandomState(1234)
    x1 = rng.uniform(-1, 1, (n, 3)) + [-10, 0, 0]
    x2 = rng.uniform(-1, 1, (n, 3)) + [10, 0, 0]
    x = np.vstack((x1, x2)) * [1, 100, 10000]
    y = [-1] * n + [1] * n
    for solver in ('svd', 'lsqr', 'eigen'):
        clf = LinearDiscriminantAnalysis(solver=solver)
        assert clf.fit(x, y).score(x, y) == 1.0, 'using covariance: %s' % solver