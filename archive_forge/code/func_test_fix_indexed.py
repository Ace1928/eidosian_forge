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
def test_fix_indexed(self):
    """Test fix variables method"""
    self.model.B = RangeSet(4)
    self.model.y = Var(self.model.B, dense=True)
    self.instance = self.model.create_instance()
    self.assertEqual(len(self.instance.y) > 0, True)
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, None)
        self.assertEqual(self.instance.y[a].fixed, False)
    self.instance.y.fix()
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, None)
        self.assertEqual(self.instance.y[a].fixed, True)
    self.instance.y.free()
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, None)
        self.assertEqual(self.instance.y[a].fixed, False)
    self.instance.y.fix(1)
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, 1)
        self.assertEqual(self.instance.y[a].fixed, True)
    self.instance.y.unfix()
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, 1)
        self.assertEqual(self.instance.y[a].fixed, False)
    self.instance.y.fix(None)
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, None)
        self.assertEqual(self.instance.y[a].fixed, True)
    self.instance.y.unfix()
    for a in self.instance.y:
        self.assertEqual(self.instance.y[a].value, None)
        self.assertEqual(self.instance.y[a].fixed, False)
    self.instance.y[1].fix()
    self.assertEqual(self.instance.y[1].value, None)
    self.assertEqual(self.instance.y[1].fixed, True)
    self.instance.y[1].free()
    self.assertEqual(self.instance.y[1].value, None)
    self.assertEqual(self.instance.y[1].fixed, False)
    self.instance.y[1].fix(value=1)
    self.assertEqual(self.instance.y[1].value, 1)
    self.assertEqual(self.instance.y[1].fixed, True)
    self.instance.y[1].unfix()
    self.assertEqual(self.instance.y[1].value, 1)
    self.assertEqual(self.instance.y[1].fixed, False)
    self.instance.y[1].fix(value=None)
    self.assertEqual(self.instance.y[1].value, None)
    self.assertEqual(self.instance.y[1].fixed, True)