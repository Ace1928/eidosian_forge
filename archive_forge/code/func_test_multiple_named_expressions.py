import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_multiple_named_expressions(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.x.value = 1
    m.y.value = 1
    m.E = pyo.Expression(expr=m.x * m.y)
    e = m.E - m.E
    derivs = reverse_ad(e)
    self.assertAlmostEqual(derivs[m.x], 0)
    self.assertAlmostEqual(derivs[m.y], 0)
    symbolic = reverse_sd(e)
    self.assertAlmostEqual(pyo.value(symbolic[m.x]), 0)
    self.assertAlmostEqual(pyo.value(symbolic[m.y]), 0)