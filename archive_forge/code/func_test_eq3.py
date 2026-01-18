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
def test_eq3(self):
    """Various checks for set equality and inequality (3)"""
    self.assertEqual(self.instance.S == self.instance.S, True)
    self.assertEqual(self.instance.S != self.instance.S, False)
    self.assertEqual(self.instance.S == self.instance.T, True)
    self.assertEqual(self.instance.T == self.instance.S, True)
    self.assertEqual(self.instance.S != self.instance.R, True)
    self.assertEqual(self.instance.R != self.instance.S, True)
    self.assertEqual(self.instance.A['A'] == self.instance.Q_a, True)
    self.assertEqual(self.instance.Q_a == self.instance.A['A'], True)
    self.assertEqual(self.instance.A['C'] == self.instance.Q_c, True)
    self.assertEqual(self.instance.Q_c == self.instance.A['C'], True)
    self.assertEqual(self.instance.A == 1.0, False)
    self.assertEqual(1.0 == self.instance.A, False)
    self.assertEqual(self.instance.A != 1.0, True)
    self.assertEqual(1.0 != self.instance.A, True)