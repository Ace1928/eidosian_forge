import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('x_array_constr', [np.array] + CSR_CONTAINERS, ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_euclidean_distances_sym(global_dtype, x_array_constr):
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    X[X < 0.8] = 0
    expected = squareform(pdist(X))
    X = x_array_constr(X)
    distances = euclidean_distances(X)
    assert_allclose(distances, expected, rtol=1e-06)
    assert distances.dtype == global_dtype