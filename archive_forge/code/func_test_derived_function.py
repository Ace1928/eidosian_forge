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
def test_derived_function(self):

    def _scalar(m):
        return 10
    dynf = types.FunctionType(_scalar.__code__, {})
    a = Initializer(dynf)
    self.assertIs(type(a), ScalarCallInitializer)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(a(None, None), 10)

    def _indexed(m, i):
        return 10 + i
    dynf = types.FunctionType(_indexed.__code__, {})
    a = Initializer(dynf)
    self.assertIs(type(a), IndexedCallInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertEqual(a(None, 5), 15)