import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_check_clustering_error():
    rng = np.random.RandomState(42)
    noise = rng.rand(500)
    wavelength = np.linspace(0.01, 1, 500) * 1e-06
    msg = 'Clustering metrics expects discrete values but received continuous values for label, and continuous values for target'
    with pytest.warns(UserWarning, match=msg):
        check_clusterings(wavelength, noise)