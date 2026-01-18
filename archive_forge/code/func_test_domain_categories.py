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
def test_domain_categories(self):
    """Test domain attribute"""
    x = Var()
    x.construct()
    self.assertEqual(x.is_integer(), False)
    self.assertEqual(x.is_binary(), False)
    self.assertEqual(x.is_continuous(), True)
    self.assertEqual(x.bounds, (None, None))
    x.domain = Integers
    self.assertEqual(x.is_integer(), True)
    self.assertEqual(x.is_binary(), False)
    self.assertEqual(x.is_continuous(), False)
    self.assertEqual(x.bounds, (None, None))
    x.domain = Binary
    self.assertEqual(x.is_integer(), True)
    self.assertEqual(x.is_binary(), True)
    self.assertEqual(x.is_continuous(), False)
    self.assertEqual(x.bounds, (0, 1))
    x.domain = RangeSet(0, 10, 0)
    self.assertEqual(x.is_integer(), False)
    self.assertEqual(x.is_binary(), False)
    self.assertEqual(x.is_continuous(), True)
    self.assertEqual(x.bounds, (0, 10))
    x.domain = RangeSet(0, 10, 1)
    self.assertEqual(x.is_integer(), True)
    self.assertEqual(x.is_binary(), False)
    self.assertEqual(x.is_continuous(), False)
    self.assertEqual(x.bounds, (0, 10))
    x.domain = RangeSet(0.5, 10, 1)
    self.assertEqual(x.is_integer(), False)
    self.assertEqual(x.is_binary(), False)
    self.assertEqual(x.is_continuous(), False)
    self.assertEqual(x.bounds, (0.5, 9.5))
    x.domain = RangeSet(0, 1, 1)
    self.assertEqual(x.is_integer(), True)
    self.assertEqual(x.is_binary(), True)
    self.assertEqual(x.is_continuous(), False)
    self.assertEqual(x.bounds, (0, 1))