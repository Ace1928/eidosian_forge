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
def test_naming(self):
    m = ConcreteModel()
    i = Set()
    self.assertEqual(str(i), 'AbstractOrderedScalarSet')
    i.construct()
    self.assertEqual(str(i), '{}')
    m.I = i
    self.assertEqual(str(i), 'I')
    j = Set(initialize=[1, 2, 3])
    self.assertEqual(str(j), 'AbstractOrderedScalarSet')
    j.construct()
    self.assertEqual(str(j), '{1, 2, 3}')
    m.J = j
    self.assertEqual(str(j), 'J')
    k = Set([1, 2, 3])
    self.assertEqual(str(k), 'IndexedSet')
    with self.assertRaisesRegex(ValueError, 'The component has not been constructed.'):
        str(k[1])
    m.K = k
    self.assertEqual(str(k), 'K')
    self.assertEqual(str(k[1]), 'K[1]')