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
def test_le2(self):
    """Various checks for set subset (2)"""
    self.assertFalse(self.instance.A < self.instance.tmpset2)
    self.assertFalse(self.instance.A <= self.instance.tmpset2)
    self.assertTrue(self.instance.A > self.instance.tmpset2)
    self.assertTrue(self.instance.A >= self.instance.tmpset2)
    self.assertTrue(self.instance.tmpset2 < self.instance.A)
    self.assertTrue(self.instance.tmpset2 <= self.instance.A)
    self.assertFalse(self.instance.tmpset2 > self.instance.A)
    self.assertFalse(self.instance.tmpset2 >= self.instance.A)