import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
def test_censure_on_rectangular_images():
    """Censure feature detector should work on 2D image of any shape."""
    rect_image = np.random.rand(300, 200)
    square_image = np.random.rand(200, 200)
    CENSURE().detect(square_image)
    CENSURE().detect(rect_image)