import functools
import pickle
import platform
import sys
import types
import pyomo.common.unittest as unittest
from pyomo.common.config import ConfigValue, ConfigList, ConfigDict
from pyomo.common.dependencies import (
from pyomo.core.base.util import flatten_tuple
from pyomo.core.base.initializer import (
from pyomo.environ import ConcreteModel, Var
def test_config_integration(self):
    c = ConfigList()
    c.add(1)
    c.add(3)
    c.add(5)
    a = Initializer(c)
    self.assertIs(type(a), ItemInitializer)
    self.assertTrue(a.contains_indices())
    self.assertEqual(list(a.indices()), [0, 1, 2])
    self.assertEqual(a(None, 0), 1)
    self.assertEqual(a(None, 1), 3)
    self.assertEqual(a(None, 2), 5)
    c = ConfigDict()
    c.declare('opt_1', ConfigValue(default=1))
    c.declare('opt_3', ConfigValue(default=3))
    c.declare('opt_5', ConfigValue(default=5))
    a = Initializer(c)
    self.assertIs(type(a), ItemInitializer)
    self.assertTrue(a.contains_indices())
    self.assertEqual(list(a.indices()), ['opt_1', 'opt_3', 'opt_5'])
    self.assertEqual(a(None, 'opt_1'), 1)
    self.assertEqual(a(None, 'opt_3'), 3)
    self.assertEqual(a(None, 'opt_5'), 5)