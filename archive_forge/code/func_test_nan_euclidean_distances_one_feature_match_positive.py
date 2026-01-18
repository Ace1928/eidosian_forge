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
@pytest.mark.parametrize('missing_value', [np.nan, -1])
def test_nan_euclidean_distances_one_feature_match_positive(missing_value):
    X = np.array([[-122.27, 648.0, missing_value, 37.85], [-122.27, missing_value, 2.34701493, missing_value]])
    dist_squared = nan_euclidean_distances(X, missing_values=missing_value, squared=True)
    assert np.all(dist_squared >= 0)
    dist = nan_euclidean_distances(X, missing_values=missing_value, squared=False)
    assert_allclose(dist, 0.0)