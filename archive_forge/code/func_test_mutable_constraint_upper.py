import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Var, Constraint, value
def test_mutable_constraint_upper(self):
    model = AbstractModel()
    model.P = Param(initialize=2.0, mutable=True)
    model.X = Var()

    def constraint_rule(m):
        return m.X <= m.P
    model.C = Constraint(rule=constraint_rule)
    instance = model.create_instance()
    self.assertEqual(value(instance.C.upper), 2.0)
    instance.P = 4.0
    self.assertEqual(value(instance.C.upper), 4.0)