import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_lower_f_accepted(self):
    assert_(self.met(res_new=self.res_new, res_old=self.res_old))