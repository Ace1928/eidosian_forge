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
@pytest.mark.parametrize('y_array_constr', [np.array] + CSR_CONTAINERS, ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_euclidean_distances_known_result(x_array_constr, y_array_constr):
    X = x_array_constr([[0]])
    Y = y_array_constr([[1], [2]])
    D = euclidean_distances(X, Y)
    assert_allclose(D, [[1.0, 2.0]])