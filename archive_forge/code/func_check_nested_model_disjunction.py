from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def check_nested_model_disjunction(self, m, bt):
    cons = bt.get_transformed_constraints(m.x, m.outer)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, -10.0 * m.outer_d1.binary_indicator_var + 0.0 * m.outer_d2.binary_indicator_var <= m.x)
    assertExpressionsEqual(self, ub.expr, 11.0 * m.outer_d1.binary_indicator_var + 0.0 * m.outer_d2.binary_indicator_var >= m.x)
    cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, -10.0 * m.outer_d1.inner_d1.binary_indicator_var - 7.0 * m.outer_d1.inner_d2.binary_indicator_var <= m.x)
    assertExpressionsEqual(self, ub.expr, 3.0 * m.outer_d1.inner_d1.binary_indicator_var + 11.0 * m.outer_d1.inner_d2.binary_indicator_var >= m.x)
    self.assertFalse(m.outer_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d1.c.active)
    self.assertFalse(m.outer_d1.inner_d2.c.active)
    self.assertFalse(m.outer_d2.c.active)