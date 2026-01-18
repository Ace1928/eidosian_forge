import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
def test_get_variables(self):
    self.M.x = Var([1, 2, 3])
    self.M.c = SOSConstraint(var=self.M.x, sos=1)
    self.assertEqual(set((id(v) for v in self.M.c.get_variables())), set((id(v) for v in self.M.x.values())))