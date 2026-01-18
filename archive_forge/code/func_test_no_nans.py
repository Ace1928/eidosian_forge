import numpy as np
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage import data, img_as_float
from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max
from skimage._shared import testing
def test_no_nans():
    """Test that `match_template` doesn't return NaN values.

    When image values are only slightly different, floating-point errors can
    cause a subtraction inside of a square root to go negative (without an
    explicit check that was added to `match_template`).
    """
    np.random.seed(1)
    image = 0.5 + 1e-09 * np.random.normal(size=(20, 20))
    template = np.ones((6, 6))
    template[:3, :] = 0
    result = match_template(image, template)
    assert not np.any(np.isnan(result))