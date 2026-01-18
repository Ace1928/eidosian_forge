import numpy as np
from numpy.testing import (assert_allclose,
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state
def test_brute_force_comparison_reverse(self):
    actual = directed_hausdorff(self.path_2, self.path_1)[0]
    expected = max(np.amin(distance.cdist(self.path_1, self.path_2), axis=0))
    assert_allclose(actual, expected)