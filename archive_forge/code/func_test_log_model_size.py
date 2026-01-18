import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_log_model_size(self):
    """Test logging functionality."""
    m = ConcreteModel()
    m.x = Var(domain=Integers)
    m.d = Disjunct()
    m.d.c = Constraint(expr=m.x == 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x == 5)
    m.disj = Disjunction(expr=[m.d2])
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.util.model_size', logging.INFO):
        log_model_size_report(m)
    expected_output = '\nactivated:\n    binary_variables: 1\n    constraints: 1\n    continuous_variables: 0\n    disjunctions: 1\n    disjuncts: 1\n    integer_variables: 1\n    nonlinear_constraints: 0\n    variables: 2\noverall:\n    binary_variables: 2\n    constraints: 2\n    continuous_variables: 0\n    disjunctions: 1\n    disjuncts: 2\n    integer_variables: 1\n    nonlinear_constraints: 0\n    variables: 3\nwarning:\n    unassociated_disjuncts: 1\n        '.strip()
    self.assertEqual(output.getvalue().strip(), expected_output)