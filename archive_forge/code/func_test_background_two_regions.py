import numpy as np
from skimage.measure import label
import skimage.measure._ccomp as ccomp
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
def test_background_two_regions(self):
    x = np.zeros((2, 3, 3), int)
    x[0] = np.array([[0, 0, 6], [0, 0, 6], [5, 5, 5]])
    x[1] = np.array([[6, 6, 0], [5, 0, 0], [0, 0, 0]])
    lb = x.copy()
    lb[0] = np.array([[BG, BG, 1], [BG, BG, 1], [2, 2, 2]])
    lb[1] = np.array([[1, 1, BG], [2, BG, BG], [BG, BG, BG]])
    res = label(x, background=0)
    assert_array_equal(res, lb)