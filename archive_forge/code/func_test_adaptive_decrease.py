import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def test_adaptive_decrease(self):
    x = 0.0
    self.takestep(x)
    self.takestep.report(True)
    for i in range(self.takestep.interval):
        self.takestep(x)
        self.takestep.report(False)
    assert_(self.ts.stepsize < self.stepsize)