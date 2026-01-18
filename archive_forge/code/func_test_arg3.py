import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, SOSConstraint, Var, Set
def test_arg3(self):
    M = ConcreteModel()
    M.v = Var()
    try:
        M.c = SOSConstraint(var=M.v)
        self.fail('Expected TypeError')
    except TypeError:
        pass