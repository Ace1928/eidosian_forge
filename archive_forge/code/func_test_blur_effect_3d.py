from numpy.testing import assert_array_equal
from skimage.color import rgb2gray
from skimage.data import astronaut, cells3d
from skimage.filters import gaussian
from skimage.measure import blur_effect
def test_blur_effect_3d():
    """Test that the blur metric works on a 3D image."""
    image_3d = cells3d()[:, 1, :, :]
    B0 = blur_effect(image_3d)
    B1 = blur_effect(gaussian(image_3d, sigma=1))
    B2 = blur_effect(gaussian(image_3d, sigma=4))
    assert 0 <= B0 < 1
    assert B0 < B1 < B2