import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
def get_dataset_for_binary_tree(random_seed, features=3):
    rng = np.random.RandomState(random_seed)
    _X = rng.rand(100, features)
    _Y = rng.rand(5, features)
    X_64 = _X.astype(dtype=np.float64, copy=False)
    Y_64 = _Y.astype(dtype=np.float64, copy=False)
    X_32 = _X.astype(dtype=np.float32, copy=False)
    Y_32 = _Y.astype(dtype=np.float32, copy=False)
    return (X_64, X_32, Y_64, Y_32)