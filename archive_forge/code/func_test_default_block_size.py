import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_default_block_size():
    image = np.arange(4 * 6).reshape(4, 6)
    out = block_reduce(image, func=np.min)
    expected = np.array([[0, 2, 4], [12, 14, 16]])
    assert_equal(expected, out)