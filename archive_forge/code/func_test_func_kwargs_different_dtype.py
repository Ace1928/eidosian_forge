import numpy as np
from skimage.measure import block_reduce
from skimage._shared import testing
from skimage._shared.testing import assert_equal
def test_func_kwargs_different_dtype():
    image = np.array([[0.45745366, 0.67479345, 0.20949775, 0.3147348], [0.7209286, 0.88915504, 0.66153409, 0.07919526], [0.04640037, 0.54008495, 0.34664343, 0.56152301], [0.58085003, 0.80144708, 0.87844473, 0.29811511]], dtype=np.float64)
    out = block_reduce(image, (2, 2), func=np.mean, func_kwargs={'dtype': np.float16})
    expected = np.array([[0.6855, 0.3164], [0.4922, 0.521]], dtype=np.float16)
    assert_equal(out, expected)
    assert out.dtype == expected.dtype