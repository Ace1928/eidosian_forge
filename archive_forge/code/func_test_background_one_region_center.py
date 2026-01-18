import numpy as np
from skimage.measure import label
import skimage.measure._ccomp as ccomp
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
def test_background_one_region_center(self):
    x = np.zeros((3, 3, 3), int)
    x[1, 1, 1] = 1
    lb = np.ones_like(x) * BG
    lb[1, 1, 1] = 1
    assert_array_equal(label(x, connectivity=1, background=0), lb)