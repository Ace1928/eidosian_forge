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
def test_issue_758(self):
    m = ConcreteModel()
    m.I = RangeSet(5)
    self.assertEqual(m.I.next(1), 2)
    self.assertEqual(m.I.next(4), 5)
    with self.assertRaisesRegex(IndexError, 'Cannot advance past the end of the Set'):
        m.I.next(5)
    self.assertEqual(m.I.prev(2), 1)
    self.assertEqual(m.I.prev(5), 4)
    with self.assertRaisesRegex(IndexError, 'Cannot advance before the beginning of the Set'):
        m.I.prev(1)
    self.assertEqual(m.I.nextw(1), 2)
    self.assertEqual(m.I.nextw(4), 5)
    self.assertEqual(m.I.nextw(5), 1)
    self.assertEqual(m.I.prevw(2), 1)
    self.assertEqual(m.I.prevw(5), 4)
    self.assertEqual(m.I.prevw(1), 5)