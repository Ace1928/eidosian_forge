import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
@pytest.mark.parametrize('shape,channel_axis', [((3, 3, 3), None), ((3, 3), -1), ((3, 3, 3, 3), -1)])
def test_hog_incorrect_dimensions(shape, channel_axis):
    img = np.zeros(shape)
    with pytest.raises(ValueError):
        feature.hog(img, channel_axis=channel_axis, block_norm='L1')