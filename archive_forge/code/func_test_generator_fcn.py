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
def test_generator_fcn(self):

    def a_init(m):
        yield 0
        yield 3
    with self.assertRaisesRegex(ValueError, 'Generator functions are not allowed'):
        a = Initializer(a_init)
    a = Initializer(a_init, allow_generators=True)
    self.assertIs(type(a), ScalarCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(list(a(None, 1)), [0, 3])

    def x_init(m, i):
        yield i
        yield (i + 1)
    a = Initializer(x_init, allow_generators=True)
    self.assertIs(type(a), IndexedCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(list(a(None, 1)), [1, 2])

    def y_init(m, i, j):
        yield j
        yield (i + 1)
    a = Initializer(y_init, allow_generators=True)
    self.assertIs(type(a), IndexedCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(list(a(None, (1, 4))), [4, 2])