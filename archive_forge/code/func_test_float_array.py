import numpy as np
from skimage.util import unique_rows
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_float_array():
    ar = np.array([[1.1, 0.0, 1.1], [0.0, 1.1, 0.0], [1.1, 0.0, 1.1]], float)
    ar_out = unique_rows(ar)
    desired_ar_out = np.array([[0.0, 1.1, 0.0], [1.1, 0.0, 1.1]], float)
    assert_equal(ar_out, desired_ar_out)