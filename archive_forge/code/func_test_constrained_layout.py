import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_constrained_layout(self):
    """Test with the constrained layout GDP model."""
    exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
    model = exfile.build_constrained_layout_model()
    model_size = build_model_size_report(model)
    self.assertEqual(model_size.activated.variables, 30)
    self.assertEqual(model_size.overall.variables, 30)
    self.assertEqual(model_size.activated.binary_variables, 18)
    self.assertEqual(model_size.overall.binary_variables, 18)
    self.assertEqual(model_size.activated.integer_variables, 0)
    self.assertEqual(model_size.overall.integer_variables, 0)
    self.assertEqual(model_size.activated.constraints, 48)
    self.assertEqual(model_size.overall.constraints, 48)
    self.assertEqual(model_size.activated.disjuncts, 18)
    self.assertEqual(model_size.overall.disjuncts, 18)
    self.assertEqual(model_size.activated.disjunctions, 6)
    self.assertEqual(model_size.overall.disjunctions, 6)