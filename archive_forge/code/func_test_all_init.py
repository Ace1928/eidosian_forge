import re
import sys
from io import StringIO
import numpy as np
import pytest
from scipy import sparse as sp
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, k_means, kmeans_plusplus
from sklearn.cluster._k_means_common import (
from sklearn.cluster._kmeans import _labels_inertia, _mini_batch_step
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS, threadpool_limits
@pytest.mark.parametrize('input_data', [X] + X_as_any_csr, ids=data_containers_ids)
@pytest.mark.parametrize('init', ['random', 'k-means++', centers, lambda X, k, random_state: centers], ids=['random', 'k-means++', 'ndarray', 'callable'])
@pytest.mark.parametrize('Estimator', [KMeans, MiniBatchKMeans])
def test_all_init(Estimator, input_data, init):
    n_init = 10 if isinstance(init, str) else 1
    km = Estimator(init=init, n_clusters=n_clusters, random_state=42, n_init=n_init).fit(input_data)
    _check_fitted_model(km)