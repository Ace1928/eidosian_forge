from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def test_evaluate_pw_function(self):
    m = self.make_ln_x_model()
    m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3])
    self.assertAlmostEqual(m.pw(1), 0)
    self.assertAlmostEqual(m.pw(2), m.f1(2))
    self.assertAlmostEqual(m.pw(3), log(3))
    self.assertAlmostEqual(m.pw(4.5), m.f2(4.5))
    self.assertAlmostEqual(m.pw(9.2), m.f3(9.2))
    self.assertAlmostEqual(m.pw(10), log(10))