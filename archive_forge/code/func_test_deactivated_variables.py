import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_deactivated_variables(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.c = Constraint(expr=m.x >= 1)
    m.c2 = Constraint(expr=m.y <= 6)
    m.c2.deactivate()
    model_size = build_model_size_report(m)
    self.assertEqual(model_size.activated.variables, 1)
    self.assertEqual(model_size.overall.variables, 2)