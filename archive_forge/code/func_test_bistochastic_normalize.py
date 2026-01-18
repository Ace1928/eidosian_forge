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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_bistochastic_normalize(global_random_seed, csr_container):
    generator = np.random.RandomState(global_random_seed)
    X = generator.rand(100, 100)
    for mat in (X, csr_container(X)):
        scaled = _bistochastic_normalize(mat)
        _do_bistochastic_test(scaled)
        if issparse(mat):
            assert issparse(scaled)