import numpy as np
import pytest
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS + [None])
def test_dtype_preserved(csr_container, global_dtype):
    """Check that centers dtype is the same as input data dtype."""
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2).astype(global_dtype, copy=False)
    if csr_container is not None:
        X[X < 0.8] = 0
        X = csr_container(X)
    km = BisectingKMeans(n_clusters=3, random_state=0)
    km.fit(X)
    assert km.cluster_centers_.dtype == global_dtype