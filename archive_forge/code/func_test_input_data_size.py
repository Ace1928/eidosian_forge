import copy
import itertools
import pickle
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from sklearn.metrics._dist_metrics import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, create_memmap_backed_data
from sklearn.utils.fixes import CSR_CONTAINERS, parse_version, sp_version
def test_input_data_size():

    def custom_metric(x, y):
        assert x.shape[0] == 3
        return np.sum((x - y) ** 2)
    rng = check_random_state(0)
    X = rng.rand(10, 3)
    pyfunc = DistanceMetric.get_metric('pyfunc', func=custom_metric)
    eucl = DistanceMetric.get_metric('euclidean')
    assert_allclose(pyfunc.pairwise(X), eucl.pairwise(X) ** 2)