import numpy as np
import numpy.testing as npt
from numpy.testing import assert_raises
from statsmodels.distributions import StepFunction, monotone_fn_inverter
from statsmodels.distributions import ECDFDiscrete
def test_StepFunctionBadShape(self):
    x = np.arange(20)
    y = np.arange(21)
    assert_raises(ValueError, StepFunction, x, y)
    x = np.zeros((2, 2))
    y = np.zeros((2, 2))
    assert_raises(ValueError, StepFunction, x, y)