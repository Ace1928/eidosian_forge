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
@pytest.mark.parametrize('batch_size', [None, 5, 7, 101])
@pytest.mark.parametrize('x_array_constr', [np.array] + CSR_CONTAINERS, ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
@pytest.mark.parametrize('y_array_constr', [np.array] + CSR_CONTAINERS, ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_euclidean_distances_upcast(batch_size, x_array_constr, y_array_constr):
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(np.float32)
    X[X < 0.8] = 0
    Y = rng.random_sample((10, 10)).astype(np.float32)
    Y[Y < 0.8] = 0
    expected = cdist(X, Y)
    X = x_array_constr(X)
    Y = y_array_constr(Y)
    distances = _euclidean_distances_upcast(X, Y=Y, batch_size=batch_size)
    distances = np.sqrt(np.maximum(distances, 0))
    assert_allclose(distances, expected, rtol=1e-06)