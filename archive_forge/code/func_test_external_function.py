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
def test_external_function(self):
    m = ConcreteModel()
    m.x = Var()
    m.sq = ExternalFunction(fgh=sum_sq)
    m.c = Constraint(expr=m.sq(m.x - 3) == 0)
    with LoggingIntercept(level=logging.DEBUG) as LOG:
        calculate_variable_from_constraint(m.x, m.c)
    self.assertAlmostEqual(value(m.x), 3, 3)
    self.assertEqual(LOG.getvalue(), 'Calculating symbolic derivative of expression failed. Reverting to numeric differentiation\n')