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
def test_mixed_ranges_issubset(self):
    i = RangeSet(0, 10, 2)
    j = SetOf([0, 1, 2, 'a'])
    k = Any
    ir, = list(i.ranges())
    jr0, jr1, jr2, jr3 = list(j.ranges())
    kr, = list(k.ranges())
    self.assertTrue(ir.issubset(ir))
    self.assertFalse(ir.issubset(jr0))
    self.assertFalse(ir.issubset(jr1))
    self.assertFalse(ir.issubset(jr3))
    self.assertTrue(ir.issubset(kr))
    self.assertTrue(jr0.issubset(ir))
    self.assertTrue(jr0.issubset(jr0))
    self.assertFalse(jr0.issubset(jr1))
    self.assertFalse(jr0.issubset(jr3))
    self.assertTrue(jr0.issubset(kr))
    self.assertFalse(jr1.issubset(ir))
    self.assertFalse(jr1.issubset(jr0))
    self.assertTrue(jr1.issubset(jr1))
    self.assertFalse(jr1.issubset(jr3))
    self.assertTrue(jr1.issubset(kr))
    self.assertFalse(jr3.issubset(ir))
    self.assertFalse(jr3.issubset(jr0))
    self.assertFalse(jr3.issubset(jr1))
    self.assertTrue(jr3.issubset(jr3))
    self.assertTrue(jr3.issubset(kr))
    self.assertFalse(kr.issubset(ir))
    self.assertFalse(kr.issubset(jr0))
    self.assertFalse(kr.issubset(jr1))
    self.assertFalse(kr.issubset(jr3))
    self.assertTrue(kr.issubset(kr))