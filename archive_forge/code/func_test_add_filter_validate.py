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
def test_add_filter_validate(self):
    m = ConcreteModel()
    m.I = Set(domain=Integers)
    self.assertIs(m.I.filter, None)
    with self.assertRaisesRegex(ValueError, 'Cannot add value 1.5 to Set I.\\n\\tThe value is not in the domain Integers'):
        m.I.add(1.5)
    self.assertTrue(m.I.add(1.0))
    self.assertIn(1, m.I)
    self.assertIn(1.0, m.I)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertFalse(m.I.add(1))
    self.assertEqual(output.getvalue(), 'Element 1 already exists in Set I; no action taken\n')
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertFalse(m.I.add((1,)))
    self.assertEqual(output.getvalue(), 'Element (1,) already exists in Set I; no action taken\n')
    m.J = Set()
    err = "Unable to insert '{}' into Set J:\\n\\tTypeError: ((unhashable type: 'dict')|('dict' objects are unhashable))"
    with self.assertRaisesRegex(TypeError, err):
        m.J.add({})
    self.assertTrue(m.J.add((1,)))
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertFalse(m.J.add(1))
    self.assertEqual(output.getvalue(), 'Element 1 already exists in Set J; no action taken\n')

    def _l_tri(model, i, j):
        self.assertIs(model, m)
        return i >= j
    m.K = Set(initialize=RangeSet(3) * RangeSet(3), filter=_l_tri)
    self.assertIsInstance(m.K.filter, IndexedCallInitializer)
    self.assertIs(m.K.filter._fcn, _l_tri)
    self.assertEqual(list(m.K), [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.K.add((0, 0)))
        self.assertFalse(m.K.add((0, 1)))
    self.assertEqual(output.getvalue(), '')
    self.assertEqual(list(m.K), [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (0, 0)])

    def _lt_3(model, i):
        self.assertIs(model, m)
        return i < 3
    m.L = Set([1, 2, 3, 4, 5], initialize=RangeSet(10), filter=_lt_3)
    self.assertEqual(len(m.L), 5)
    self.assertEqual(list(m.L[1]), [1, 2])
    self.assertEqual(list(m.L[5]), [1, 2])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.L[2].add(0))
        self.assertFalse(m.L[2].add(100))
    self.assertEqual(output.getvalue(), '')
    self.assertEqual(list(m.L[2]), [1, 2, 0])
    m = ConcreteModel()

    def _validate(model, i, j):
        self.assertIs(model, m)
        if i + j < 2:
            return True
        if i - j > 2:
            return False
        raise RuntimeError('Bogus value')
    m.I = Set(validate=_validate)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.I.add((0, 1)))
        self.assertEqual(output.getvalue(), '')
        with self.assertRaisesRegex(ValueError, 'The value=\\(4, 1\\) violates the validation rule of Set I'):
            m.I.add((4, 1))
        self.assertEqual(output.getvalue(), '')
        with self.assertRaisesRegex(RuntimeError, 'Bogus value'):
            m.I.add((2, 2))
    self.assertEqual(output.getvalue(), "Exception raised while validating element '(2, 2)' for Set I\n")
    m.J = Set([(0, 0), (2, 2)], validate=_validate)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        self.assertTrue(m.J[2, 2].add((0, 1)))
        self.assertEqual(output.getvalue(), '')
        with self.assertRaisesRegex(ValueError, 'The value=\\(4, 1\\) violates the validation rule of Set J\\[0,0\\]'):
            m.J[0, 0].add((4, 1))
        self.assertEqual(output.getvalue(), '')
        with self.assertRaisesRegex(RuntimeError, 'Bogus value'):
            m.J[2, 2].add((2, 2))
    self.assertEqual(output.getvalue(), "Exception raised while validating element '(2, 2)' for Set J[2,2]\n")