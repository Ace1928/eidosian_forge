from numpy.testing import assert_array_equal
from skimage.color import rgb2gray
from skimage.data import astronaut, cells3d
from skimage.filters import gaussian
from skimage.measure import blur_effect
def test_blur_effect_channel_axis():
    """Test that passing an RGB image is equivalent to passing its grayscale
    version.
    """
    image = astronaut()
    B0 = blur_effect(image, channel_axis=-1)
    B1 = blur_effect(rgb2gray(image))
    B0_arr = blur_effect(image, channel_axis=-1, reduce_func=None)
    B1_arr = blur_effect(rgb2gray(image), reduce_func=None)
    assert 0 <= B0 < 1
    assert B0 == B1
    assert_array_equal(B0_arr, B1_arr)