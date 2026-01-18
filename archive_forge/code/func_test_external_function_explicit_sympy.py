import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.errors import IterationLimitError
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.sympy_tools import sympy_available
@unittest.skipUnless(sympy_available, 'test expects that sympy is available')
def test_external_function_explicit_sympy(self):
    m = ConcreteModel()
    m.x = Var()
    m.sq = ExternalFunction(fgh=sum_sq)
    m.c = Constraint(expr=m.sq(m.x - 3) == 0)
    with self.assertRaisesRegex(TypeError, "Expressions containing external functions are not convertible to sympy expressions \\(found 'f\\(x0 - 3"):
        calculate_variable_from_constraint(m.x, m.c, diff_mode=differentiate.Modes.sympy)