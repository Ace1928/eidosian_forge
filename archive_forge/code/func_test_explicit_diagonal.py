import numpy as np
import pytest
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.neighbors._base import _is_sorted_by_data
from sklearn.utils._testing import assert_array_equal
def test_explicit_diagonal():
    n_neighbors = 5
    n_samples_fit, n_samples_transform, n_features = (20, 18, 10)
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples_fit, n_features)
    X2 = rng.randn(n_samples_transform, n_features)
    nnt = KNeighborsTransformer(n_neighbors=n_neighbors)
    Xt = nnt.fit_transform(X)
    assert _has_explicit_diagonal(Xt)
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)
    Xt = nnt.transform(X)
    assert _has_explicit_diagonal(Xt)
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)
    X2t = nnt.transform(X2)
    assert not _has_explicit_diagonal(X2t)