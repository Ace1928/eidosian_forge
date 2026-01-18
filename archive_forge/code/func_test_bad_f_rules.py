import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
def test_bad_f_rules(self):
    model = ConcreteModel()
    model.range = Var()
    model.x = Var(bounds=(-1, 1))
    args = (model.range, model.x)
    keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
    model.con = Piecewise(*args, **keywords)
    try:
        del keywords['f_rule']
        model.con1 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized without a proper f_rule keyword.')
    try:
        keywords['f_rule'] = None
        model.con2 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized without a proper f_rule keyword.')
    try:
        keywords['f_rule'] = model.x
        model.con3 = Piecewise(*args, **keywords)
    except Exception:
        pass
    else:
        self.fail('Piecewise should fail when initialized without a proper f_rule keyword.')