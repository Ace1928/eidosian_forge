import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, BiclusterMixin
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from sklearn.cluster._bicluster import (
from sklearn.datasets import make_biclusters, make_checkerboard
from sklearn.metrics import consensus_score, v_measure_score
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_fit_best_piecewise(global_random_seed):
    model = SpectralBiclustering(random_state=global_random_seed)
    vectors = np.array([[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3], [0, 1, 2, 3, 4, 5]])
    best = model._fit_best_piecewise(vectors, n_best=2, n_clusters=2)
    assert_array_equal(best, vectors[:2])