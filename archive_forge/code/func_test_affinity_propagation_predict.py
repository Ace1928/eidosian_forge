import warnings
import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_affinity_propagation_predict(global_random_seed, global_dtype):
    af = AffinityPropagation(affinity='euclidean', random_state=global_random_seed)
    X_ = X.astype(global_dtype, copy=False)
    labels = af.fit_predict(X_)
    labels2 = af.predict(X_)
    assert_array_equal(labels, labels2)