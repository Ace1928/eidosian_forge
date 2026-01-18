import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available
def approx_deriv(expr, wrt, delta=0.001):
    numerator = 0
    wrt.value += 2 * delta
    numerator -= pyo.value(expr)
    wrt.value -= delta
    numerator += 8 * pyo.value(expr)
    wrt.value -= 2 * delta
    numerator -= 8 * pyo.value(expr)
    wrt.value -= delta
    numerator += pyo.value(expr)
    wrt.value += 2 * delta
    return numerator / (12 * delta)