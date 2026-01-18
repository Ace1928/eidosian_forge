import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
@pytest.mark.parametrize('metric', itertools.chain(METRICS, BOOLEAN_METRICS))
def test_kernel_density_numerical_consistency(global_random_seed, metric):
    X_64, X_32, Y_64, Y_32 = get_dataset_for_binary_tree(random_seed=global_random_seed)
    metric_params = METRICS.get(metric, {})
    bt_64 = BallTree64(X_64, leaf_size=1, metric=metric, **metric_params)
    bt_32 = BallTree32(X_32, leaf_size=1, metric=metric, **metric_params)
    kernel = 'gaussian'
    h = 0.1
    density64 = bt_64.kernel_density(Y_64, h=h, kernel=kernel, breadth_first=True)
    density32 = bt_32.kernel_density(Y_32, h=h, kernel=kernel, breadth_first=True)
    assert_allclose(density64, density32, rtol=1e-05)
    assert density64.dtype == np.float64
    assert density32.dtype == np.float32