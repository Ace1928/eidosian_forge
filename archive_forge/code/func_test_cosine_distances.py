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
def test_cosine_distances():
    rng = np.random.RandomState(1337)
    x = np.abs(rng.rand(910))
    XA = np.vstack([x, x])
    D = cosine_distances(XA)
    assert_allclose(D, [[0.0, 0.0], [0.0, 0.0]], atol=1e-10)
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)
    assert_allclose(D[np.diag_indices_from(D)], [0.0, 0.0])
    XB = np.vstack([x, -x])
    D2 = cosine_distances(XB)
    assert np.all(D2 >= 0.0)
    assert np.all(D2 <= 2.0)
    assert_allclose(D2, [[0.0, 2.0], [2.0, 0.0]])
    X = np.abs(rng.rand(1000, 5000))
    D = cosine_distances(X)
    assert_allclose(D[np.diag_indices_from(D)], [0.0] * D.shape[0])
    assert np.all(D >= 0.0)
    assert np.all(D <= 2.0)