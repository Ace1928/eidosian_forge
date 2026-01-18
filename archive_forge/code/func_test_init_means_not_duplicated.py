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
@pytest.mark.parametrize('init_params', ['random', 'random_from_data', 'k-means++', 'kmeans'])
def test_init_means_not_duplicated(init_params, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    rand_data = RandomData(rng, scale=5)
    n_components = rand_data.n_components
    X = rand_data.X['full']
    gmm = GaussianMixture(n_components=n_components, init_params=init_params, random_state=rng, max_iter=0)
    gmm.fit(X)
    means = gmm.means_
    for i_mean, j_mean in itertools.combinations(means, r=2):
        assert not np.allclose(i_mean, j_mean)