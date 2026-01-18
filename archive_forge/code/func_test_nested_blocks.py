import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_nested_blocks(self):
    """Test with nested blocks."""
    m = ConcreteModel()
    m.b = Block()
    m.inactive_b = Block()
    m.inactive_b.deactivate()
    m.b.x = Var()
    m.b.x2 = Var(domain=Binary)
    m.b.x3 = Var(domain=Integers)
    m.inactive_b.x = Var()
    m.b.c = Constraint(expr=m.b.x == m.b.x2)
    m.inactive_b.c = Constraint(expr=m.b.x == 1)
    m.inactive_b.c2 = Constraint(expr=m.inactive_b.x == 15)
    model_size = build_model_size_report(m)
    self.assertEqual(model_size.activated.variables, 2)
    self.assertEqual(model_size.overall.variables, 4)
    self.assertEqual(model_size.activated.binary_variables, 1)
    self.assertEqual(model_size.overall.binary_variables, 1)
    self.assertEqual(model_size.activated.integer_variables, 0)
    self.assertEqual(model_size.overall.integer_variables, 1)
    self.assertEqual(model_size.activated.constraints, 1)
    self.assertEqual(model_size.overall.constraints, 3)
    self.assertEqual(model_size.activated.disjuncts, 0)
    self.assertEqual(model_size.overall.disjuncts, 0)
    self.assertEqual(model_size.activated.disjunctions, 0)
    self.assertEqual(model_size.overall.disjunctions, 0)