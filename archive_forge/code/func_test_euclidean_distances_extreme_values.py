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
@pytest.mark.parametrize('dtype, eps, rtol', [(np.float32, 0.0001, 1e-05), pytest.param(np.float64, 1e-08, 0.99, marks=pytest.mark.xfail(reason='failing due to lack of precision'))])
@pytest.mark.parametrize('dim', [1, 1000000])
def test_euclidean_distances_extreme_values(dtype, eps, rtol, dim):
    X = np.array([[1.0] * dim], dtype=dtype)
    Y = np.array([[1.0 + eps] * dim], dtype=dtype)
    distances = euclidean_distances(X, Y)
    expected = cdist(X, Y)
    assert_allclose(distances, expected, rtol=1e-05)