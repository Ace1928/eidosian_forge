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
def test_0based_initialize_with_bad_dict(self):
    """Test initialize option with a dictionary of subkeys"""
    self.model.x = VarList(initialize={0: 1.3, 1: 2.3}, starting_index=0)
    self.instance = self.model.create_instance()
    self.assertEqual(self.instance.x[0].value, 1.3)
    self.assertEqual(self.instance.x[1].value, 2.3)
    self.instance.x[0] = 1
    self.instance.x[1] = 2
    self.assertEqual(self.instance.x[0].value, 1)
    self.assertEqual(self.instance.x[1].value, 2)
    self.instance.x.add()
    self.assertEqual(list(self.instance.x.keys()), [0, 1, 2])