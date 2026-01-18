import functools
import warnings
from typing import Any, List
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
def make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=None, sparse_format='csr'):
    """Make some random data with uniformly located non zero entries with
    Gaussian distributed values; `sparse_format` can be `"csr"` (default) or
    `None` (in which case a dense array is returned).
    """
    rng = np.random.RandomState(random_state)
    data_coo = coo_container((rng.randn(n_nonzeros), (rng.randint(n_samples, size=n_nonzeros), rng.randint(n_features, size=n_nonzeros))), shape=(n_samples, n_features))
    if sparse_format is not None:
        return data_coo.asformat(sparse_format)
    else:
        return data_coo.toarray()