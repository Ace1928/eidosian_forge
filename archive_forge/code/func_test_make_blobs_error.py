import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_blobs_error():
    n_samples = [20, 20, 20]
    centers = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cluster_stds = np.array([0.05, 0.2, 0.4])
    wrong_centers_msg = re.escape(f'Length of `n_samples` not consistent with number of centers. Got n_samples = {n_samples} and centers = {centers[:-1]}')
    with pytest.raises(ValueError, match=wrong_centers_msg):
        make_blobs(n_samples, centers=centers[:-1])
    wrong_std_msg = re.escape(f'Length of `clusters_std` not consistent with number of centers. Got centers = {centers} and cluster_std = {cluster_stds[:-1]}')
    with pytest.raises(ValueError, match=wrong_std_msg):
        make_blobs(n_samples, centers=centers, cluster_std=cluster_stds[:-1])
    wrong_type_msg = 'Parameter `centers` must be array-like. Got {!r} instead'.format(3)
    with pytest.raises(ValueError, match=wrong_type_msg):
        make_blobs(n_samples, centers=3)