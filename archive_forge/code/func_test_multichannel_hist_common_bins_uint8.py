import warnings
import numpy as np
import pytest
from numpy.testing import (
from packaging.version import Version
from skimage import data
from skimage import exposure
from skimage import util
from skimage.color import rgb2gray
from skimage.exposure.exposure import intensity_range
from skimage.util.dtype import dtype_range
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('source_range', ['dtype', 'image'])
@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float64])
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_multichannel_hist_common_bins_uint8(dtype, source_range, channel_axis):
    """Check that all channels use the same binning."""
    shape = (5, 5)
    channel_size = shape[0] * shape[1]
    imin, imax = dtype_range[dtype]
    im = np.stack((np.full(shape, imin, dtype=dtype), np.full(shape, imax, dtype=dtype)), axis=channel_axis)
    frequencies, bin_centers = exposure.histogram(im, source_range=source_range, channel_axis=channel_axis)
    if np.issubdtype(dtype, np.integer):
        assert_array_equal(bin_centers, np.arange(imin, imax + 1))
    assert frequencies[0][0] == channel_size
    assert frequencies[0][-1] == 0
    assert frequencies[1][0] == 0
    assert frequencies[1][-1] == channel_size