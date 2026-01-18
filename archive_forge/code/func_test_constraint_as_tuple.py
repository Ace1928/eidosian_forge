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
def test_constraint_as_tuple(self):
    m = ConcreteModel()
    m.x = Var()
    m.p = Param(initialize=15, mutable=True)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, 5 * m.x == 5, diff_mode=mode)
        self.assertEqual(value(m.x), 1)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, (5 * m.x, 10), diff_mode=mode)
        self.assertEqual(value(m.x), 2)
    for mode in all_diff_modes:
        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, (15, 5 * m.x, m.p), diff_mode=mode)
        self.assertEqual(value(m.x), 3)
    with self.assertRaisesRegex(ValueError, "Constraint 'tuple' is a Ranged Inequality with a variable upper bound."):
        calculate_variable_from_constraint(m.x, (15, 5 * m.x, m.x))