import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_count_mean_with_matrix(self):
    m = masked_array(np.matrix([[1, 2], [3, 4]]), mask=np.zeros((2, 2)))
    assert_equal(m.count(axis=0).shape, (1, 2))
    assert_equal(m.count(axis=1).shape, (2, 1))
    assert_equal(m.mean(axis=0), [[2.0, 3.0]])
    assert_equal(m.mean(axis=1), [[1.5], [3.5]])