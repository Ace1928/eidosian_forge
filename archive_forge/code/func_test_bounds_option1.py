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
def test_bounds_option1(self):
    """Test bounds option"""

    def x_bounds(model, i, j):
        return (-1.0 * (i + j), 1.0 * (i + j))
    self.model.x = Var(self.model.A, self.model.A, bounds=x_bounds)
    self.instance = self.model.create_instance()
    self.assertEqual(value(self.instance.x[1, 1].lb), -2.0)
    self.assertEqual(value(self.instance.x[1, 2].ub), 3.0)