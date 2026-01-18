import numpy as np
from skimage.util import crop
from skimage._shared.testing import assert_array_equal, assert_equal
def test_multi_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2), (2, 1)))
    assert_array_equal(out[0], [7, 8])
    assert_array_equal(out[-1], [32, 33])
    assert_equal(out.shape, (6, 2))