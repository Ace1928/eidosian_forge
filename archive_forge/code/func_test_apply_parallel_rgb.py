import numpy as np
from skimage._shared.testing import assert_array_almost_equal, assert_equal
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel
import pytest
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('chunks', (None, (128, 128, 3)))
@pytest.mark.parametrize('depth', (0, 8, (8, 8, 0)))
def test_apply_parallel_rgb(depth, chunks, dtype):
    cat = data.chelsea().astype(dtype) / 255.0
    func = color.rgb2ycbcr
    cat_ycbcr_expected = func(cat)
    cat_ycbcr = apply_parallel(func, cat, chunks=chunks, depth=depth, dtype=dtype, channel_axis=-1)
    assert_equal(cat_ycbcr.dtype, cat.dtype)
    assert_array_almost_equal(cat_ycbcr_expected, cat_ycbcr)