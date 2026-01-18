import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_nine_process(self):
    """Test with the nine process problem model."""
    exfile = import_file(join(exdir, 'nine_process', 'small_process.py'))
    simple_model = exfile.build_model()
    simple_model_size = build_model_size_report(simple_model)
    self.assertEqual(simple_model_size.overall.variables, 34)
    self.assertEqual(simple_model_size.activated.variables, 34)