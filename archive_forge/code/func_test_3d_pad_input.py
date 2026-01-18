import numpy as np
from skimage._shared.testing import assert_almost_equal, assert_equal
from skimage import data, img_as_float
from skimage.morphology import diamond
from skimage.feature import match_template, peak_local_max
from skimage._shared import testing
def test_3d_pad_input():
    np.random.seed(1)
    template = np.random.rand(3, 3, 3)
    image = np.zeros((12, 12, 12))
    image[3:6, 5:8, 4:7] = template
    result = match_template(image, template, pad_input=True)
    assert_equal(result.shape, (12, 12, 12))
    assert_equal(np.unravel_index(result.argmax(), result.shape), (4, 6, 5))