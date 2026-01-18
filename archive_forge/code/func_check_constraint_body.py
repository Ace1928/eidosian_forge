import pyomo.common.unittest as unittest
from pyomo.core import Constraint, BooleanVar, SortComponents
from pyomo.gdp.basic_step import apply_basic_step
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
from pyomo.common.fileutils import import_file
from os.path import abspath, dirname, normpath, join
def check_constraint_body(self, m, constraint, constant):
    self.assertIsNone(constraint.lower)
    self.assertEqual(constraint.upper, 0)
    repn = generate_standard_repn(constraint.body)
    self.assertEqual(repn.constant, constant)
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, m.a, -1)
    ct.check_linear_coef(self, repn, m.x, 1)