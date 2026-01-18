import numpy as np
import itertools
from skimage import (
from skimage.util.dtype import _convert
from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize
def test_float_conversion_dtype():
    """Test any conversion from a float dtype to an other."""
    x = np.array([-1, 1])
    dtype_combin = np.array(np.meshgrid(float_dtype_list, float_dtype_list)).T.reshape(-1, 2)
    for dtype_in, dtype_out in dtype_combin:
        x = x.astype(dtype_in)
        y = _convert(x, dtype_out)
        assert y.dtype == np.dtype(dtype_out)