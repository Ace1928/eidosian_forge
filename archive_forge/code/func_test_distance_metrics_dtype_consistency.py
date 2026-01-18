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
@pytest.mark.parametrize('metric_param_grid', METRICS_DEFAULT_PARAMS, ids=lambda params: params[0])
def test_distance_metrics_dtype_consistency(metric_param_grid):
    metric, param_grid = metric_param_grid
    keys = param_grid.keys()
    rtol = 1e-05
    for vals in itertools.product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        dm64 = DistanceMetric.get_metric(metric, np.float64, **kwargs)
        dm32 = DistanceMetric.get_metric(metric, np.float32, **kwargs)
        D64 = dm64.pairwise(X64)
        D32 = dm32.pairwise(X32)
        assert D64.dtype == np.float64
        assert D32.dtype == np.float32
        assert_allclose(D64, D32, rtol=rtol)
        D64 = dm64.pairwise(X64, Y64)
        D32 = dm32.pairwise(X32, Y32)
        assert_allclose(D64, D32, rtol=rtol)