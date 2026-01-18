import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_bad_bound_type(self):
    model = ConcreteModel()
    model.range = Var()
    model.x = Var(bounds=(-1, 1))
    args = (model.range, model.x)
    keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
    model.con = Piecewise(*args, **keywords)
    try:
        keywords['pw_constr_type'] = 1.0
        model.con1 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized with invalid bound type.')
    try:
        del keywords['pw_constr_type']
        model.con1 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized with invalid bound type.')