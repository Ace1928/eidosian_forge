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
@pytest.mark.parametrize('X', [np.array([[np.inf, 0]]), np.array([[0, -np.inf]])])
@pytest.mark.parametrize('Y', [np.array([[np.inf, 0]]), np.array([[0, -np.inf]]), None])
def test_nan_euclidean_distances_infinite_values(X, Y):
    with pytest.raises(ValueError) as excinfo:
        nan_euclidean_distances(X, Y=Y)
    exp_msg = "Input contains infinity or a value too large for dtype('float64')."
    assert exp_msg == str(excinfo.value)