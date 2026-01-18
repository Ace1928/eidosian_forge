import re
import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
from pyomo.scripting.pyomo_main import main
from pyomo.core import (
from pyomo.common.tee import capture_output
from io import StringIO
def test_construct2(self):
    model = AbstractModel()
    model.a = Set(initialize=[1, 2, 3])
    model.A = Param(initialize=1)
    model.B = Param(model.a)
    model.x = Var(initialize=1, within=Reals, dense=True)
    model.y = Var(model.a, initialize=1, within=Reals, dense=True)
    model.obj = Objective(rule=lambda model: model.x + model.y[1])
    model.obj2 = Objective(model.a, rule=lambda model, i: i + model.x + model.y[1])
    model.con = Constraint(rule=rule1)
    model.con2 = Constraint(model.a, rule=rule2)
    instance = model.create_instance()
    expr = instance.x + 1
    OUTPUT = open(join(currdir, 'display2.out'), 'w')
    display(instance, ostream=OUTPUT)
    display(instance.obj, ostream=OUTPUT)
    display(instance.x, ostream=OUTPUT)
    display(instance.con, ostream=OUTPUT)
    OUTPUT.write(expr.to_string())
    model = AbstractModel()
    instance = model.create_instance()
    display(instance, ostream=OUTPUT)
    OUTPUT.close()
    try:
        display(None)
        self.fail('test_construct - expected TypeError')
    except TypeError:
        pass
    _out, _txt = (join(currdir, 'display2.out'), join(currdir, 'display2.txt'))
    self.assertTrue(cmp(_out, _txt), msg='Files %s and %s differ' % (_out, _txt))