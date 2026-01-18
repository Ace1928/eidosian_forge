import numpy as np
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology._convex_hull import possible_hull
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from skimage._shared._warnings import expected_warnings
def test_consistent_2d_3d_hulls(images2d3d):
    image, image3d = images2d3d
    chimage = convex_hull_image(image)
    chimage[8, 0] = True
    chimage3d = convex_hull_image(image3d)
    assert_array_equal(chimage3d[1], chimage)