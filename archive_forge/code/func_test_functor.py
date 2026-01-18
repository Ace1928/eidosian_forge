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
def test_functor(self):

    class InitScalar(object):

        def __init__(self, val):
            self.val = val

        def __call__(self, m):
            return self.val
    a = Initializer(InitScalar(10))
    self.assertIs(type(a), ScalarCallInitializer)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(a(None, None), 10)

    class InitIndexed(object):

        def __init__(self, val):
            self.val = val

        def __call__(self, m, i):
            return self.val + i
    a = Initializer(InitIndexed(10))
    self.assertIs(type(a), IndexedCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(a(None, 5), 15)