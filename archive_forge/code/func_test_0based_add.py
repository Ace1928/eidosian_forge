import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_0based_add(self):
    m = ConcreteModel()
    m.x = Var()
    m.c = ConstraintList(starting_index=0)
    m.c.add(m.x <= 0)
    self.assertEqual(list(m.c.keys()), [0])
    m.c.add(m.x >= 0)
    self.assertEqual(list(m.c.keys()), [0, 1])