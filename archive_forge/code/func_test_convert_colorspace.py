import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
@pytest.mark.parametrize('channel_axis', [0, -1])
def test_convert_colorspace(self, channel_axis):
    colspaces = ['HSV', 'RGB CIE', 'XYZ', 'YCbCr', 'YPbPr', 'YDbDr']
    colfuncs_from = [hsv2rgb, rgbcie2rgb, xyz2rgb, ycbcr2rgb, ypbpr2rgb, ydbdr2rgb]
    colfuncs_to = [rgb2hsv, rgb2rgbcie, rgb2xyz, rgb2ycbcr, rgb2ypbpr, rgb2ydbdr]
    colbars_array = np.moveaxis(self.colbars_array, source=-1, destination=channel_axis)
    kw = dict(channel_axis=channel_axis)
    assert_almost_equal(convert_colorspace(colbars_array, 'RGB', 'RGB', **kw), colbars_array)
    for i, space in enumerate(colspaces):
        gt = colfuncs_from[i](colbars_array, **kw)
        assert_almost_equal(convert_colorspace(colbars_array, space, 'RGB', **kw), gt)
        gt = colfuncs_to[i](colbars_array, **kw)
        assert_almost_equal(convert_colorspace(colbars_array, 'RGB', space, **kw), gt)
    with pytest.raises(ValueError):
        convert_colorspace(self.colbars_array, 'nokey', 'XYZ')
    with pytest.raises(ValueError):
        convert_colorspace(self.colbars_array, 'RGB', 'nokey')