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
def test_initialize_option(self):
    """Test initialize option"""
    self.model.x = Var(self.model.A, self.model.A, initialize={(1, 1): 1.3, (2, 2): 2.3})
    self.instance = self.model.create_instance()
    self.assertEqual(self.instance.x[1, 1].value, 1.3)
    self.assertEqual(self.instance.x[2, 2].value, 2.3)
    try:
        value(self.instance.x[1, 2])
        self.fail('Expected ValueError')
    except ValueError:
        pass