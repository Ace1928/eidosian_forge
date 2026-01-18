import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
def test_goodvalues(self):
    z = np.array((-1.0, 0.0, 1.0))
    res = np.isinf(z) == 0
    assert_all(np.all(res, axis=0))