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
def test_initialize9(self):
    self.model.A = Set(initialize=range(0, 3))

    @set_options(domain=Integers)
    def B_index(model):
        return [i / 2.0 for i in model.A]

    def B_init(model, i, j):
        if j:
            return range(int(i), int(2 + i))
        return []
    self.model.B = Set(B_index, [True, False], initialize=B_init)
    try:
        self.instance = self.model.create_instance()
        self.fail('Expected ValueError because B_index returns invalid set values')
    except ValueError:
        pass