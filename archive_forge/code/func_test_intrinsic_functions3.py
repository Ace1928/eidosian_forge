import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_intrinsic_functions3(self):
    m = ConcreteModel()
    m.x = Var()
    e = differentiate(exp(2 * m.x), wrt=m.x)
    self.assertEqual(s(e), s(2.0 * exp(2.0 * m.x)))