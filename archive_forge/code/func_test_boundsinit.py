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
def test_boundsinit(self):
    a = BoundsInitializer(5, default_step=1)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(5))
    a = BoundsInitializer((0, 5), default_step=1)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(0, 5))
    a = BoundsInitializer((0, 5, 2))
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(0, 5, 2))
    a = BoundsInitializer(())
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(None, None, 0))
    a = BoundsInitializer(5)
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(1, 5, 0))
    a = BoundsInitializer((0, 5))
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(0, 5, 0))
    a = BoundsInitializer((0, 5, 2))
    self.assertTrue(a.constant())
    self.assertFalse(a.verified)
    s = a(None, None)
    self.assertEqual(s, RangeSet(0, 5, 2))
    a = BoundsInitializer({1: 5}, default_step=1)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    s = a(None, 1)
    self.assertEqual(s, RangeSet(5))
    a = BoundsInitializer({1: (0, 5)}, default_step=1)
    self.assertFalse(a.constant())
    self.assertFalse(a.verified)
    s = a(None, 1)
    self.assertEqual(s, RangeSet(0, 5))