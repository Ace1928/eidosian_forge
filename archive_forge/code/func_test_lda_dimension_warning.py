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
@pytest.mark.parametrize('n_features', [3, 5])
@pytest.mark.parametrize('n_classes', [5, 3])
def test_lda_dimension_warning(n_classes, n_features):
    rng = check_random_state(0)
    n_samples = 10
    X = rng.randn(n_samples, n_features)
    y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    max_components = min(n_features, n_classes - 1)
    for n_components in [max_components - 1, None, max_components]:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)
    for n_components in [max_components + 1, max(n_features, n_classes - 1) + 1]:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        msg = 'n_components cannot be larger than '
        with pytest.raises(ValueError, match=msg):
            lda.fit(X, y)