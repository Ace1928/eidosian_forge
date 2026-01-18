import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_log_ndtr_moderate_le8(self):
    x = np.array([-0.75, -0.25, 0, 0.5, 1.5, 2.5, 3, 4, 5, 7, 8])
    expected = np.array([-1.4844482299196562, -0.9130617648111351, -0.6931471805599453, -0.3689464152886564, -0.06914345561223398, -0.006229025485860002, -0.0013508099647481938, -3.167174337748927e-05, -2.866516129637636e-07, -1.279812543886654e-12, -6.220960574271786e-16])
    y = sc.log_ndtr(x)
    assert_allclose(y, expected, rtol=1e-14)