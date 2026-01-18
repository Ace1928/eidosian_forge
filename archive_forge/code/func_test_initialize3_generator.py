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
def test_initialize3_generator(self):
    self.model.A = Set(initialize=lambda m: ((i, j) for i in range(0, 3) for j in range(1, 4) if (i + j) % 2 == 0))
    self.instance = self.model.create_instance()
    self.assertEqual(len(self.instance.A), 4)
    m = ConcreteModel()
    m.A = Set(initialize=((i, j) for i in range(0, 3) for j in range(1, 4) if (i + j) % 2 == 0))
    self.assertEqual(len(m.A), 4)