import numpy as np
from skimage.measure import approximate_polygon, subdivide_polygon
from skimage.measure._polygon import _SUBDIVISION_MASKS
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
def test_approximate_polygon():
    out = approximate_polygon(square, 0.1)
    assert_array_equal(out, square[(0, 3, 6, 9, 12), :])
    out = approximate_polygon(square, 2.2)
    assert_array_equal(out, square[(0, 6, 12), :])
    out = approximate_polygon(square[(0, 1, 3, 4, 5, 6, 7, 9, 11, 12), :], 0.1)
    assert_array_equal(out, square[(0, 3, 6, 9, 12), :])
    out = approximate_polygon(square, -1)
    assert_array_equal(out, square)
    out = approximate_polygon(square, 0)
    assert_array_equal(out, square)