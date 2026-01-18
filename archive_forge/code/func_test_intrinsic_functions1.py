import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_intrinsic_functions1(self):
    m = ConcreteModel()
    m.x = Var()
    e = differentiate(log(m.x), wrt=m.x)
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(m.x ** (-1.0)))