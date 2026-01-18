import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_activate_deactivate_nonindexed(self):
    model = ConcreteModel()
    model.y = Var()
    model.x = Var(bounds=(-1, 1))
    args = (model.y, model.x)
    keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
    model.c = Piecewise(*args, **keywords)
    self.assertTrue(len(model.c.component_map(Constraint)) > 0)
    self.assertTrue(len(model.c.component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, True)
    model.c.deactivate()
    self.assertTrue(len(model.c.component_map(Constraint)) > 0)
    self.assertTrue(len(model.c.component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, False)
    model.c.activate()
    self.assertTrue(len(model.c.component_map(Constraint)) > 0)
    self.assertTrue(len(model.c.component_map(Constraint, active=True)) > 0)
    self.assertEqual(model.c.active, True)