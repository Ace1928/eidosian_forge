from numpy.testing import assert_array_equal
from skimage.color import rgb2gray
from skimage.data import astronaut, cells3d
from skimage.filters import gaussian
from skimage.measure import blur_effect
def test_blur_effect_h_size():
    """Test that the blur metric decreases with increasing size of the
    re-blurring filter.
    """
    image = astronaut()
    B0 = blur_effect(image, h_size=3, channel_axis=-1)
    B1 = blur_effect(image, channel_axis=-1)
    B2 = blur_effect(image, h_size=30, channel_axis=-1)
    assert 0 <= B0 < 1
    assert B0 > B1 > B2