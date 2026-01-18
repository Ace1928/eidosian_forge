import numpy as np
from numpy.testing import (assert_allclose,
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state
def test_2d_data_forward(self):
    actual = directed_hausdorff(self.path_1[..., :2], self.path_2[..., :2])[0]
    expected = max(np.amin(distance.cdist(self.path_1[..., :2], self.path_2[..., :2]), axis=1))
    assert_allclose(actual, expected)