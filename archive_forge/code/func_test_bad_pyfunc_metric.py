import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_equal
from sklearn.neighbors._ball_tree import BallTree, BallTree32, BallTree64
from sklearn.utils import check_random_state
from sklearn.utils._testing import _convert_container
from sklearn.utils.validation import check_array
@pytest.mark.parametrize('BallTreeImplementation', BALL_TREE_CLASSES)
def test_bad_pyfunc_metric(BallTreeImplementation):

    def wrong_returned_value(x, y):
        return '1'

    def one_arg_func(x):
        return 1.0
    X = np.ones((5, 2))
    msg = 'Custom distance function must accept two vectors and return a float.'
    with pytest.raises(TypeError, match=msg):
        BallTreeImplementation(X, metric=wrong_returned_value)
    msg = 'takes 1 positional argument but 2 were given'
    with pytest.raises(TypeError, match=msg):
        BallTreeImplementation(X, metric=one_arg_func)