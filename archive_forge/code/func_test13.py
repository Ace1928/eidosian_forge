import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
def test13(self):
    I = {0: [1, 2], 1: [2, 3]}
    M = ConcreteModel()
    M.x = Var([1, 2, 3], dense=True)
    M.c = SOSConstraint([0, 1], var=M.x, index=I, sos=1)
    self.assertEqual(set(((v.name, w) for v, w in M.c[0].get_items())), set(((M.x[i].name, i) for i in I[0])))
    self.assertEqual(set(((v.name, w) for v, w in M.c[1].get_items())), set(((M.x[i].name, i - 1) for i in I[1])))