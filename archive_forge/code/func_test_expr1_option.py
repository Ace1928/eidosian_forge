import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_expr1_option(self):
    """Test expr option"""
    model = ConcreteModel()
    model.B = RangeSet(1, 4)
    model.x = Var(model.B, initialize=2)
    ans = 0
    for i in model.B:
        ans = ans + model.x[i]
    model.obj = Objective(expr=ans)
    self.assertEqual(model.obj(), 8)
    self.assertEqual(value(model.obj), 8)
    self.assertEqual(value(model.obj._data[None]), 8)