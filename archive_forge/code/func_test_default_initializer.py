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
def test_default_initializer(self):
    a = Initializer({1: 5})
    d = DefaultInitializer(a, None, KeyError)
    self.assertFalse(d.constant())
    self.assertTrue(d.contains_indices())
    self.assertEqual(list(d.indices()), [1])
    self.assertEqual(d(None, 1), 5)
    self.assertIsNone(d(None, 2))

    def rule(m, i):
        if i == 0:
            return 10
        elif i == 1:
            raise KeyError('key')
        elif i == 2:
            raise TypeError('type')
        else:
            raise RuntimeError('runtime')
    a = Initializer(rule)
    d = DefaultInitializer(a, 100, (KeyError, RuntimeError))
    self.assertFalse(d.constant())
    self.assertFalse(d.contains_indices())
    self.assertEqual(d(None, 0), 10)
    self.assertEqual(d(None, 1), 100)
    with self.assertRaisesRegex(TypeError, 'type'):
        d(None, 2)
    self.assertEqual(d(None, 3), 100)