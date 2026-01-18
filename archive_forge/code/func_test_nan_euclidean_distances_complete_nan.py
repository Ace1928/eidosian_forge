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
def test_nan_euclidean_distances_complete_nan(missing_value):
    X = np.array([[missing_value, missing_value], [0, 1]])
    exp_dist = np.array([[np.nan, np.nan], [np.nan, 0]])
    dist = nan_euclidean_distances(X, missing_values=missing_value)
    assert_allclose(exp_dist, dist)
    dist = nan_euclidean_distances(X, X.copy(), missing_values=missing_value)
    assert_allclose(exp_dist, dist)