import numpy as np
from skimage.measure import label
import skimage.measure._ccomp as ccomp
from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
def test_connectivity_1_vs_2(self):
    x = np.zeros((2, 2, 2), int)
    x[0, 1, 1] = 1
    x[1, 0, 0] = 1
    label1 = x.copy()
    label1[1, 0, 0] = 2
    assert_array_equal(label(x, connectivity=1), label1)
    assert_array_equal(label(x, connectivity=3), x)