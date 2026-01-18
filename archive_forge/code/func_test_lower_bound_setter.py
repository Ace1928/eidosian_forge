import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_lower_bound_setter(self):
    m = ConcreteModel()
    m.x = Var()
    self.assertIsNone(m.x.lb)
    m.x.lb = 1
    self.assertEqual(m.x.lb, 1)
    m.x.lower = 2
    self.assertEqual(m.x.lb, 2)
    m.x.setlb(3)
    self.assertEqual(m.x.lb, 3)
    m.y = Var([1])
    self.assertIsNone(m.y[1].lb)
    m.y[1].lb = 1
    self.assertEqual(m.y[1].lb, 1)
    m.y[1].lower = 2
    self.assertEqual(m.y[1].lb, 2)
    m.y[1].setlb(3)
    self.assertEqual(m.y[1].lb, 3)