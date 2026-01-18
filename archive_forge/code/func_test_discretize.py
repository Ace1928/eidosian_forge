import pickle
import re
import numpy as np
import pytest
from scipy.linalg import LinAlgError
from sklearn.cluster import SpectralClustering, spectral_clustering
from sklearn.cluster._spectral import cluster_qr, discretize
from sklearn.datasets import make_blobs
from sklearn.feature_extraction import img_to_graph
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import kernel_metrics, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
@pytest.mark.parametrize('n_samples', [50, 100, 150, 500])
def test_discretize(n_samples, coo_container):
    random_state = np.random.RandomState(seed=8)
    for n_class in range(2, 10):
        y_true = random_state.randint(0, n_class + 1, n_samples)
        y_true = np.array(y_true, float)
        y_indicator = coo_container((np.ones(n_samples), (np.arange(n_samples), y_true)), shape=(n_samples, n_class + 1))
        y_true_noisy = y_indicator.toarray() + 0.1 * random_state.randn(n_samples, n_class + 1)
        y_pred = discretize(y_true_noisy, random_state=random_state)
        assert adjusted_rand_score(y_true, y_pred) > 0.8