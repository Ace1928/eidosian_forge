import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_mutable_param_expr(self):
    """Test expr option with a single mutable param"""
    model = ConcreteModel()
    model.p = Param(initialize=1.0, mutable=True)
    model.obj = Objective(expr=model.p)
    self.assertEqual(model.obj(), 1.0)
    self.assertEqual(value(model.obj), 1.0)
    self.assertEqual(value(model.obj._data[None]), 1.0)