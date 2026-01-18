import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_indexed_with_nonindexed_vars(self):
    model = ConcreteModel()
    model.range1 = Var()
    model.x = Var(bounds=(-1, 1))
    args = ([1], model.range1, model.x)
    keywords = {'pw_pts': {1: [-1, 0, 1]}, 'pw_constr_type': 'EQ', 'f_rule': lambda model, i, x: x ** 2}
    model.con1 = Piecewise(*args, **keywords)
    model.range2 = Var([1])
    model.y = Var([1], bounds=(-1, 1))
    args = ([1], model.range2, model.y)
    model.con2 = Piecewise(*args, **keywords)
    args = ([1], model.range2, model.y[1])
    model.con3 = Piecewise(*args, **keywords)