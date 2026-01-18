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
def check_linear_func_constraints(self, m, mbm, Ms=None):
    if Ms is None:
        Ms = self.get_Ms(m)
    cons = mbm.get_transformed_constraints(m.d1.func)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.assertEqual(value(lower.upper), 0)
    self.assertIsNone(lower.lower)
    repn = generate_standard_repn(lower.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 5)
    self.assertEqual(repn.constant, 0)
    check_linear_coef(self, repn, m.x1, -1)
    check_linear_coef(self, repn, m.x2, -1)
    check_linear_coef(self, repn, m.d, 1)
    check_linear_coef(self, repn, m.d2.binary_indicator_var, Ms[m.d1.func, m.d2][0])
    check_linear_coef(self, repn, m.d3.binary_indicator_var, Ms[m.d1.func, m.d3][0])
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.assertEqual(value(upper.upper), 0)
    self.assertIsNone(upper.lower)
    repn = generate_standard_repn(upper.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 5)
    self.assertEqual(repn.constant, 0)
    check_linear_coef(self, repn, m.x1, 1)
    check_linear_coef(self, repn, m.x2, 1)
    check_linear_coef(self, repn, m.d, -1)
    check_linear_coef(self, repn, m.d2.binary_indicator_var, -Ms[m.d1.func, m.d2][1])
    check_linear_coef(self, repn, m.d3.binary_indicator_var, -Ms[m.d1.func, m.d3][1])
    cons = mbm.get_transformed_constraints(m.d2.func)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.assertEqual(value(lower.upper), 0)
    self.assertIsNone(lower.lower)
    repn = generate_standard_repn(lower.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 5)
    self.assertEqual(repn.constant, -7)
    check_linear_coef(self, repn, m.x1, -2)
    check_linear_coef(self, repn, m.x2, -4)
    check_linear_coef(self, repn, m.d, 1)
    check_linear_coef(self, repn, m.d1.binary_indicator_var, Ms[m.d2.func, m.d1][0])
    check_linear_coef(self, repn, m.d3.binary_indicator_var, Ms[m.d2.func, m.d3][0])
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.assertEqual(value(upper.upper), 0)
    self.assertIsNone(upper.lower)
    repn = generate_standard_repn(upper.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 5)
    self.assertEqual(repn.constant, 7)
    check_linear_coef(self, repn, m.x1, 2)
    check_linear_coef(self, repn, m.x2, 4)
    check_linear_coef(self, repn, m.d, -1)
    check_linear_coef(self, repn, m.d1.binary_indicator_var, -Ms[m.d2.func, m.d1][1])
    check_linear_coef(self, repn, m.d3.binary_indicator_var, -Ms[m.d2.func, m.d3][1])
    cons = mbm.get_transformed_constraints(m.d3.func)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.assertEqual(value(lower.upper), 0)
    self.assertIsNone(lower.lower)
    repn = generate_standard_repn(lower.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 5)
    self.assertEqual(repn.constant, 3)
    check_linear_coef(self, repn, m.x1, -1)
    check_linear_coef(self, repn, m.x2, 5)
    check_linear_coef(self, repn, m.d, 1)
    check_linear_coef(self, repn, m.d1.binary_indicator_var, Ms[m.d3.func, m.d1][0])
    check_linear_coef(self, repn, m.d2.binary_indicator_var, Ms[m.d3.func, m.d2][0])
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.assertEqual(value(upper.upper), 0)
    self.assertIsNone(upper.lower)
    repn = generate_standard_repn(upper.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 5)
    self.assertEqual(repn.constant, -3)
    check_linear_coef(self, repn, m.x1, 1)
    check_linear_coef(self, repn, m.x2, -5)
    check_linear_coef(self, repn, m.d, -1)
    check_linear_coef(self, repn, m.d1.binary_indicator_var, -Ms[m.d3.func, m.d1][1])
    check_linear_coef(self, repn, m.d2.binary_indicator_var, -Ms[m.d3.func, m.d2][1])