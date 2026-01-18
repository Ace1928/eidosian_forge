import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_euclidean_param_defaults():
    tf = EuclideanTransform(translation=(5, 5))
    assert np.array(tf)[0, 1] == 0
    tf = EuclideanTransform(translation=(4, 5, 9), dimensionality=3)
    assert_equal(np.array(tf)[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], 0)
    with pytest.raises(ValueError):
        _ = EuclideanTransform(translation=(5, 6, 7, 8), dimensionality=4)
    with pytest.raises(ValueError):
        _ = EuclideanTransform(rotation=(4, 8), dimensionality=3)
    tf = EuclideanTransform(rotation=np.pi * np.arange(3), dimensionality=3)
    assert_equal(np.array(tf)[:-1, 3], 0)