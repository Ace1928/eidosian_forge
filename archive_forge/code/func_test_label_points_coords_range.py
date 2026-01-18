import numpy as np
from skimage._shared import testing
from skimage._shared.testing import assert_equal
from skimage.util._label import label_points
def test_label_points_coords_range():
    coords, output_shape = (np.array([[0, 0], [5, 5]]), (5, 5))
    with testing.raises(IndexError):
        label_points(coords, output_shape)