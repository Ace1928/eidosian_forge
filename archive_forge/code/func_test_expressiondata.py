import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def test_expressiondata(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=3)
    m.e = pyo.Expression(expr=m.x * 2)

    @m.Expression([1, 2])
    def e2(m, i):
        if i == 1:
            return m.x + 4
        else:
            return m.x ** 2
    m.o = pyo.Objective(expr=m.e + 1 + m.e2[1] + m.e2[2])
    derivs = reverse_ad(m.o.expr)
    symbolic = reverse_sd(m.o.expr)
    self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol)