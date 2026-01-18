import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler
def test_linear_model_row_col_order(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var([1, 2, 3])
    m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
    m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)
    repn = LinearStandardFormCompiler().write(m, column_order=[m.y[3], m.y[2], m.x, m.y[1]], row_order=[m.d, m.c])
    self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
    self.assertTrue(np.all(repn.A == np.array([[4, 0, 1], [0, -1, -2]])))
    self.assertTrue(np.all(repn.rhs == np.array([5, -3])))