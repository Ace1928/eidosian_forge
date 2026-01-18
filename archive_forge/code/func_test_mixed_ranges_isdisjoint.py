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
def test_mixed_ranges_isdisjoint(self):
    i = RangeSet(0, 10, 2)
    j = SetOf([0, 1, 2, 'a'])
    k = Any
    ir = list(i.ranges())
    self.assertEqual(ir, [NR(0, 10, 2)])
    self.assertEqual(str(ir), '[[0:10:2]]')
    ir = ir[0]
    jr = list(j.ranges())
    self.assertEqual(jr, [NR(0, 0, 0), NR(1, 1, 0), NR(2, 2, 0), NNR('a')])
    self.assertEqual(str(jr), '[[0], [1], [2], {a}]')
    jr0, jr1, jr2, jr3 = jr
    kr = list(k.ranges())
    self.assertEqual(kr, [AnyRange()])
    self.assertEqual(str(kr), '[[*]]')
    kr = kr[0]
    self.assertFalse(ir.isdisjoint(ir))
    self.assertFalse(ir.isdisjoint(jr0))
    self.assertTrue(ir.isdisjoint(jr1))
    self.assertTrue(ir.isdisjoint(jr3))
    self.assertFalse(ir.isdisjoint(kr))
    self.assertFalse(jr0.isdisjoint(ir))
    self.assertFalse(jr0.isdisjoint(jr0))
    self.assertTrue(jr0.isdisjoint(jr1))
    self.assertTrue(jr0.isdisjoint(jr3))
    self.assertFalse(jr0.isdisjoint(kr))
    self.assertTrue(jr1.isdisjoint(ir))
    self.assertTrue(jr1.isdisjoint(jr0))
    self.assertFalse(jr1.isdisjoint(jr1))
    self.assertTrue(jr1.isdisjoint(jr3))
    self.assertFalse(jr1.isdisjoint(kr))
    self.assertTrue(jr3.isdisjoint(ir))
    self.assertTrue(jr3.isdisjoint(jr0))
    self.assertTrue(jr3.isdisjoint(jr1))
    self.assertFalse(jr3.isdisjoint(jr3))
    self.assertFalse(jr3.isdisjoint(kr))
    self.assertFalse(kr.isdisjoint(ir))
    self.assertFalse(kr.isdisjoint(jr0))
    self.assertFalse(kr.isdisjoint(jr1))
    self.assertFalse(kr.isdisjoint(jr3))
    self.assertFalse(kr.isdisjoint(kr))