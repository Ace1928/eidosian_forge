import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def test_inequality_comparison_fails(self):
    x = RealSet()
    y = RealSet()
    self.assertFalse(x < y)
    self.assertTrue(x <= y)
    self.assertFalse(x > y)
    self.assertTrue(x >= y)