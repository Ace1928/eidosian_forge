import numpy as np
from numpy.testing import (assert_allclose,
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state
def test_invalid_dimensions(self):
    rng = np.random.default_rng(189048172503940875434364128139223470523)
    A = rng.random((3, 2))
    B = rng.random((3, 5))
    msg = 'need to have the same number of columns'
    with pytest.raises(ValueError, match=msg):
        directed_hausdorff(A, B)