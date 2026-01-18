from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
@unittest.skipUnless(numpy_available, 'numpy is not available')
def test_evaluate_pw_linear_function(self):
    m = self.make_model()

    def g1(x1, x2):
        return 3 * x1 + 5 * x2 - 4

    def g2(x1, x2):
        return 3 * x1 + 11 * x2 - 28
    m.pw = PiecewiseLinearFunction(simplices=self.simplices, linear_functions=[g1, g1, g2, g2])
    for x1, x2 in m.pw._points:
        self.assertAlmostEqual(m.pw(x1, x2), m.g(x1, x2))
    self.assertAlmostEqual(m.pw(1, 3), g1(1, 3))
    self.assertAlmostEqual(m.pw(2.5, 6), g2(2.5, 6))
    self.assertAlmostEqual(m.pw(0.2, 4.3), g2(0.2, 4.3))