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
@unittest.skipUnless(pandas_available, 'Pandas is not installed')
def test_initializer_initializer(self):
    d = {'col1': [1, 2, 4], 'col2': [10, 20, 40]}
    df = pd.DataFrame(data=d)
    a = Initializer(DataFrameInitializer(df, 'col2'))
    self.assertIs(type(a), DataFrameInitializer)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    self.assertTrue(a.contains_indices())
    self.assertEqual(list(a.indices()), [0, 1, 2])
    self.assertEqual(a(None, 0), 10)
    self.assertEqual(a(None, 1), 20)
    self.assertEqual(a(None, 2), 40)