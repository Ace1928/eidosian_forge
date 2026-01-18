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
def test_bounds_option2(self):
    """Test bounds option"""
    self.model.x = Var(self.model.A, self.model.A, bounds=(-1.0, 1.0))
    self.instance = self.model.create_instance()
    self.assertEqual(value(self.instance.x[1, 1].lb), -1.0)
    self.assertEqual(value(self.instance.x[1, 1].ub), 1.0)