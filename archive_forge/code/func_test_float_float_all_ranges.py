import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_float_float_all_ranges():
    arr_in = np.array([[-10.0, 10.0, 1e+20]], dtype=np.float32)
    np.testing.assert_array_equal(img_as_float(arr_in), arr_in)