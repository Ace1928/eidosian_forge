import copy
import itertools
import re
import sys
import warnings
from io import StringIO
from unittest.mock import Mock
import numpy as np
import pytest
from scipy import linalg, stats
import sklearn
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_spd_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
def test_check_means():
    rng = np.random.RandomState(0)
    rand_data = RandomData(rng)
    n_components, n_features = (rand_data.n_components, rand_data.n_features)
    X = rand_data.X['full']
    g = GaussianMixture(n_components=n_components)
    means_bad_shape = rng.rand(n_components + 1, n_features)
    g.means_init = means_bad_shape
    msg = "The parameter 'means' should have the shape of "
    with pytest.raises(ValueError, match=msg):
        g.fit(X)
    means = rand_data.means
    g.means_init = means
    g.fit(X)
    assert_array_equal(means, g.means_init)