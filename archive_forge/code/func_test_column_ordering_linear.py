import os
import random
from ..lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Block, ComponentMap
def test_column_ordering_linear(self):
    model = ConcreteModel()
    model.a = Var()
    model.b = Var()
    model.c = Var()
    terms = [model.a, model.b, model.c]
    model.obj = Objective(expr=self._gen_expression(terms))
    model.con = Constraint(expr=self._gen_expression(terms) <= 1)
    column_order = ComponentMap()
    column_order[model.a] = 2
    column_order[model.b] = 1
    column_order[model.c] = 0
    self._check_baseline(model, column_order=column_order)