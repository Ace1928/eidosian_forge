import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_invalid_block_size():
    image = np.arange(4 * 6).reshape(4, 6)
    with testing.raises(ValueError):
        block_reduce(image, [1, 2, 3])
    with testing.raises(ValueError):
        block_reduce(image, [1, 0.5])