import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_RealSet_IntegerSet(self):
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        a = SetModule.RealSet()
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    self.assertEqual(a, Reals)
    self.assertIsNot(a, Reals)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        a = SetModule.RealSet(bounds=(1, 3))
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    self.assertEqual(a.bounds(), (1, 3))
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        a = SetModule.IntegerSet()
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    self.assertEqual(a, Integers)
    self.assertIsNot(a, Integers)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        a = SetModule.IntegerSet(bounds=(1, 3))
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    self.assertEqual(a.bounds(), (1, 3))
    self.assertEqual(list(a), [1, 2, 3])
    m = ConcreteModel()
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m.x = Var(within=SetModule.RealSet)
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m.y = Var(within=SetModule.RealSet())
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m.z = Var(within=SetModule.RealSet(bounds=(0, None)))
    self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
    with self.assertRaisesRegex(RuntimeError, "Unexpected keyword arguments: \\{'foo': 5\\}"):
        IntegerSet(foo=5)