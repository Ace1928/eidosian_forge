import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_conlist_skip(self):
    model = ConcreteModel()
    model.x = Var()
    model.c = ObjectiveList()
    self.assertTrue(1 not in model.c)
    self.assertEqual(len(model.c), 0)
    model.c.add(Objective.Skip)
    self.assertTrue(1 not in model.c)
    self.assertEqual(len(model.c), 0)
    model.c.add(model.x + 1)
    self.assertTrue(1 not in model.c)
    self.assertTrue(2 in model.c)
    self.assertEqual(len(model.c), 1)