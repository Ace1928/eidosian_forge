import numpy as np
import numpy.testing as npt
from numpy.testing import assert_raises
from statsmodels.distributions import StepFunction, monotone_fn_inverter
from statsmodels.distributions import ECDFDiscrete
def test_ecdf_discrete(self):
    x = [3, 3, 1, 4]
    e = ECDFDiscrete(x)
    npt.assert_array_equal(e.x, [-np.inf, 1, 3, 4])
    npt.assert_array_equal(e.y, [0, 0.25, 0.75, 1])
    e1 = ECDFDiscrete([3.5, 3.5, 1.5, 1, 4])
    e2 = ECDFDiscrete([3.5, 1.5, 1, 4], freq_weights=[2, 1, 1, 1])
    npt.assert_array_equal(e1.x, e2.x)
    npt.assert_array_equal(e1.y, e2.y)