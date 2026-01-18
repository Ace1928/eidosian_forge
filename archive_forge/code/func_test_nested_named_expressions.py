import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_nested_named_expressions(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.23)
    m.y = pyo.Var(initialize=0.88)
    m.a = pyo.Expression(expr=(m.x + 1) ** 2)
    m.b = pyo.Expression(expr=3 * (m.a + m.y))
    e = 2 * m.a + 2 * m.b + 2 * m.b + 2 * m.a
    derivs = reverse_ad(e)
    symbolic = reverse_sd(e)
    self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
    self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
    self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
    self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)