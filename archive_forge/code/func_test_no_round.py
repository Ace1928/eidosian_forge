from skimage.draw import line_nd
from skimage._shared.testing import assert_equal
def test_no_round():
    coords = line_nd((0.5, 0), (2.5, 0), integer=False, endpoint=True)
    assert_equal(coords, [[0.5, 1.5, 2.5], [0, 0, 0]])