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
def test_no_normalize_index(self):
    try:
        _oldFlatten = normalize_index.flatten
        normalize_index.flatten = False
        m = ConcreteModel()
        m.I = Set()
        self.assertIs(m.I._dimen, UnknownSetDimen)
        self.assertTrue(m.I.add((1, (2, 3))))
        self.assertIs(m.I._dimen, 2)
        self.assertNotIn(((1, 2), 3), m.I)
        self.assertIn((1, (2, 3)), m.I)
        self.assertNotIn((1, 2, 3), m.I)
        m.J = Set()
        self.assertTrue(m.J.add(1))
        self.assertIn(1, m.J)
        self.assertNotIn((1,), m.J)
        self.assertTrue(m.J.add((1,)))
        self.assertIn(1, m.J)
        self.assertIn((1,), m.J)
        self.assertTrue(m.J.add((2,)))
        self.assertNotIn(2, m.J)
        self.assertIn((2,), m.J)
        normalize_index.flatten = True
        m = ConcreteModel()
        m.I = Set()
        self.assertIs(m.I._dimen, UnknownSetDimen)
        m.I.add((1, (2, 3)))
        self.assertIs(m.I._dimen, 3)
        self.assertIn(((1, 2), 3), m.I)
        self.assertIn((1, (2, 3)), m.I)
        self.assertIn((1, 2, 3), m.I)
        m.J = Set()
        self.assertTrue(m.J.add(1))
        self.assertIn(1, m.J)
        self.assertIn((1,), m.J)
        self.assertFalse(m.J.add((1,)))
        self.assertIn(1, m.J)
        self.assertIn((1,), m.J)
        self.assertTrue(m.J.add((2,)))
        self.assertIn(2, m.J)
        self.assertIn((2,), m.J)
    finally:
        normalize_index.flatten = _oldFlatten