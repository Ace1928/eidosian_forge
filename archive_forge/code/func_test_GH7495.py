import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_GH7495(self):
    met = Metropolis(2)
    res_new = OptimizeResult(success=True, fun=0.0)
    res_old = OptimizeResult(success=True, fun=2000)
    with np.errstate(over='raise'):
        met.accept_reject(res_new=res_new, res_old=res_old)