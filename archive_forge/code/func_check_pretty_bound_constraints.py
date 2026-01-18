from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def check_pretty_bound_constraints(self, cons, var, bounds, lb):
    self.assertEqual(value(cons.upper), 0)
    self.assertIsNone(cons.lower)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), len(bounds) + 1)
    self.assertEqual(repn.constant, 0)
    if lb:
        check_linear_coef(self, repn, var, -1)
        for disj, bnd in bounds.items():
            check_linear_coef(self, repn, disj.binary_indicator_var, bnd)
    else:
        check_linear_coef(self, repn, var, 1)
        for disj, bnd in bounds.items():
            check_linear_coef(self, repn, disj.binary_indicator_var, -bnd)