import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_float_warning(self, dtype):
    image = self.image.astype(dtype)
    msg = 'Applying `local_binary_pattern` to floating-point images'
    with expected_warnings([msg]):
        lbp = local_binary_pattern(image, 8, 1, 'ror')
    ref = np.array([[0, 127, 0, 255, 3, 255], [31, 0, 5, 51, 1, 7], [119, 255, 3, 127, 0, 63], [3, 1, 31, 63, 31, 0], [255, 1, 255, 95, 0, 127], [3, 5, 0, 255, 1, 3]])
    np.testing.assert_array_equal(lbp, ref)