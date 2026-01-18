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
@pytest.mark.parametrize('squared', [True, False])
def test_nan_euclidean_distances_equal_to_euclidean_distance(squared):
    rng = np.random.RandomState(1337)
    X = rng.randn(3, 4)
    Y = rng.randn(4, 4)
    normal_distance = euclidean_distances(X, Y=Y, squared=squared)
    nan_distance = nan_euclidean_distances(X, Y=Y, squared=squared)
    assert_allclose(normal_distance, nan_distance)