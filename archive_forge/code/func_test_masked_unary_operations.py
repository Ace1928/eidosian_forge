import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_masked_unary_operations(self):
    x, mx = self.data
    with np.errstate(divide='ignore'):
        assert_(isinstance(log(mx), MMatrix))
        assert_equal(log(x), np.log(x))