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
def test_initialize8(self):
    self.model.A = Set(initialize=range(0, 3))

    def B_index(model):
        return [(i, i + 1, i * i) for i in model.A]

    def B_init(model, i, ii, iii, j):
        if j:
            return range(i, 2 + i)
        return []
    self.model.B = Set(B_index, [True, False], initialize=B_init)
    instance = self.model.create_instance()
    self.assertEqual(len(instance.B), 6)
    self.assertEqual(instance.B[0, 1, 0, False], [])
    self.assertEqual(instance.B[0, 1, 0, True], [0, 1])
    self.assertEqual(instance.B[1, 2, 1, False], [])
    self.assertEqual(instance.B[1, 2, 1, True], [1, 2])
    self.assertEqual(instance.B[2, 3, 4, False], [])
    self.assertEqual(instance.B[2, 3, 4, True], [2, 3])