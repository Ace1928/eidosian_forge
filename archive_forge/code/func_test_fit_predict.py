import numpy as np
import pytest
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS + [None])
def test_fit_predict(csr_container):
    """Check if labels from fit(X) method are same as from fit(X).predict(X)."""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    if csr_container is not None:
        X[X < 0.8] = 0
        X = csr_container(X)
    bisect_means = BisectingKMeans(n_clusters=3, random_state=0)
    bisect_means.fit(X)
    assert_array_equal(bisect_means.labels_, bisect_means.predict(X))