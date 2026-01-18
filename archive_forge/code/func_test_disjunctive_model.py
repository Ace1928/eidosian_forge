import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_disjunctive_model(self):
    from pyomo.gdp.tests.models import makeNestedDisjunctions
    m = makeNestedDisjunctions()
    model_size = build_model_size_report(m)
    self.assertEqual(model_size.activated.variables, 10)
    self.assertEqual(model_size.overall.variables, 10)
    self.assertEqual(model_size.activated.binary_variables, 7)
    self.assertEqual(model_size.overall.binary_variables, 7)
    self.assertEqual(model_size.activated.integer_variables, 0)
    self.assertEqual(model_size.overall.integer_variables, 0)
    self.assertEqual(model_size.activated.constraints, 6)
    self.assertEqual(model_size.overall.constraints, 6)
    self.assertEqual(model_size.activated.disjuncts, 7)
    self.assertEqual(model_size.overall.disjuncts, 7)
    self.assertEqual(model_size.activated.disjunctions, 3)
    self.assertEqual(model_size.overall.disjunctions, 3)