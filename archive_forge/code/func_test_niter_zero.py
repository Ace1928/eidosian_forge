import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_niter_zero(self):
    i = 0
    basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs, niter=0, disp=self.disp)