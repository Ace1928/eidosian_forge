import numpy as np
from skimage._shared.testing import assert_array_almost_equal, assert_equal
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel
import pytest
@pytest.mark.parametrize('chunks', (None, (128, 256), 'ndim'))
@pytest.mark.parametrize('depth', (0, 8, (8, 16), 'ndim'))
@pytest.mark.parametrize('channel_axis', (0, 1, 2, -1, -2, -3))
def test_apply_parallel_rgb_channel_axis(depth, chunks, channel_axis):
    """Test channel_axis combinations.

    For depth and chunks, test in three ways:
    1.) scalar (to be applied over all axes)
    2.) tuple of length ``image.ndim - 1`` corresponding to spatial axes
    3.) tuple of length ``image.ndim`` corresponding to all axes
    """
    cat = img_as_float(data.chelsea())
    func = color.rgb2ycbcr
    cat_ycbcr_expected = func(cat, channel_axis=-1)
    cat = np.moveaxis(cat, -1, channel_axis)
    if chunks == 'ndim':
        chunks = [128, 128]
        chunks.insert(channel_axis % cat.ndim, cat.shape[channel_axis])
    if depth == 'ndim':
        depth = [8, 8]
        depth.insert(channel_axis % cat.ndim, 0)
    cat_ycbcr = apply_parallel(func, cat, chunks=chunks, depth=depth, dtype=cat.dtype, channel_axis=channel_axis, extra_keywords=dict(channel_axis=channel_axis))
    cat_ycbcr = np.moveaxis(cat_ycbcr, channel_axis, -1)
    assert_array_almost_equal(cat_ycbcr_expected, cat_ycbcr)