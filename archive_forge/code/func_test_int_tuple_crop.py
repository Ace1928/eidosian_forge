import numpy as np
from skimage.util import crop
from skimage._shared.testing import assert_array_equal, assert_equal
def test_int_tuple_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, (1,))
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert_equal(out.shape, (7, 3))