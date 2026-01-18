from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def test_indexed_pw_linear_function_linear_funcs_and_simplices(self):
    m = self.make_ln_x_model()
    m.z = Var([1, 2], bounds=(-10, 10))

    def silly_simplex_rule(m, i):
        return [(1, 3), (3, 6), (6, 10)]

    def h1(x):
        return 4 * x - 3

    def h2(x):
        return 9 * x - 18

    def h3(x):
        return 16 * x - 60

    def silly_linear_func_rule(m, i):
        return [h1, h2, h3]
    m.pw = PiecewiseLinearFunction([1, 2], simplices=silly_simplex_rule, linear_functions=silly_linear_func_rule)
    self.check_x_squared_approx(m.pw[1], m.z[1])
    self.check_x_squared_approx(m.pw[2], m.z[2])