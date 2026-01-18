import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def test_arg_errors(self):
    m = models.makeTwoTermDisj()
    m.simple = Constraint(expr=m.x <= m.a + 1)
    with self.assertRaisesRegex(ValueError, 'apply_basic_step only accepts a list containing Disjunctions or Constraints'):
        apply_basic_step([m.disjunction, m.simple, m.x])
    with self.assertRaisesRegex(ValueError, 'apply_basic_step: argument list must contain at least one Disjunction'):
        apply_basic_step([m.simple, m.simple])