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
def test_fix_all(self):
    """Test fix all variables method"""
    self.model.B = RangeSet(4)
    self.model.x = Var()
    self.model.y = Var(self.model.B, dense=True)
    self.instance = self.model.create_instance()
    self.instance.fix_all_vars()
    self.assertEqual(self.instance.x.fixed, True)
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].fixed, True)