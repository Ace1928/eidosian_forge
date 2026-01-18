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
def test_Any(self):
    self.assertIn(0, Any)
    self.assertIn(1.5, Any)
    (self.assertIn(100, Any),)
    (self.assertIn(-100, Any),)
    self.assertIn('A', Any)
    self.assertIn(None, Any)
    self.assertFalse(Any.isdiscrete())
    self.assertFalse(Any.isfinite())
    self.assertEqual(Any.dim(), 0)
    self.assertIs(Any.index_set(), UnindexedComponent_set)
    with self.assertRaisesRegex(TypeError, ".*'Any' has no len"):
        len(Any)
    with self.assertRaisesRegex(TypeError, "'GlobalSet' object is not iterable \\(non-finite Set 'Any' is not iterable\\)"):
        list(Any)
    self.assertEqual(list(Any.ranges()), [AnyRange()])
    self.assertEqual(Any.bounds(), (None, None))
    self.assertEqual(Any.dimen, None)
    tmp = _AnySet()
    self.assertFalse(tmp.isdiscrete())
    self.assertFalse(tmp.isfinite())
    self.assertEqual(Any, tmp)
    tmp.clear()
    self.assertEqual(Any, tmp)
    self.assertEqual(tmp.domain, Any)
    self.assertEqual(str(Any), 'Any')
    self.assertEqual(str(tmp), '_AnySet')
    b = ConcreteModel()
    b.tmp = tmp
    self.assertEqual(str(tmp), 'tmp')