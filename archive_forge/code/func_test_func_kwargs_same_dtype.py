import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_func_kwargs_same_dtype():
    image = np.array([[97, 123, 173, 227], [217, 241, 221, 214], [211, 11, 170, 53], [214, 205, 101, 57]], dtype=np.uint8)
    out = block_reduce(image, (2, 2), func=np.mean, func_kwargs={'dtype': np.uint8})
    expected = np.array([[41, 16], [32, 31]], dtype=np.uint8)
    assert_equal(out, expected)
    assert out.dtype == expected.dtype