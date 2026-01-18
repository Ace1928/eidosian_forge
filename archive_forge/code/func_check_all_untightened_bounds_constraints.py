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
def check_all_untightened_bounds_constraints(self, m, mbm):
    cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.check_untightened_bounds_constraint(lower, m.x1, m.d1, m.disjunction, {m.d2: 0.65, m.d3: 2}, lower=0.5)
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.check_untightened_bounds_constraint(upper, m.x1, m.d1, m.disjunction, {m.d2: 3, m.d3: 10}, upper=2)
    cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.check_untightened_bounds_constraint(lower, m.x2, m.d1, m.disjunction, {m.d2: 3, m.d3: 0.55}, lower=0.75)
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.check_untightened_bounds_constraint(upper, m.x2, m.d1, m.disjunction, {m.d2: 10, m.d3: 1}, upper=3)
    cons = mbm.get_transformed_constraints(m.d2.x1_bounds)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.check_untightened_bounds_constraint(lower, m.x1, m.d2, m.disjunction, {m.d1: 0.5, m.d3: 2}, lower=0.65)
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.check_untightened_bounds_constraint(upper, m.x1, m.d2, m.disjunction, {m.d1: 2, m.d3: 10}, upper=3)
    cons = mbm.get_transformed_constraints(m.d2.x2_bounds)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.check_untightened_bounds_constraint(lower, m.x2, m.d2, m.disjunction, {m.d1: 0.75, m.d3: 0.55}, lower=3)
    upper = cons[1]
    self.check_untightened_bounds_constraint(upper, m.x2, m.d2, m.disjunction, {m.d1: 3, m.d3: 1}, upper=10)
    cons = mbm.get_transformed_constraints(m.d3.x1_bounds)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    self.check_untightened_bounds_constraint(lower, m.x1, m.d3, m.disjunction, {m.d1: 0.5, m.d2: 0.65}, lower=2)
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.check_untightened_bounds_constraint(upper, m.x1, m.d3, m.disjunction, {m.d1: 2, m.d2: 3}, upper=10)
    cons = mbm.get_transformed_constraints(m.d3.x2_bounds)
    self.assertEqual(len(cons), 2)
    lower = cons[0]
    check_obj_in_active_tree(self, lower)
    self.check_untightened_bounds_constraint(lower, m.x2, m.d3, m.disjunction, {m.d1: 0.75, m.d2: 3}, lower=0.55)
    upper = cons[1]
    check_obj_in_active_tree(self, upper)
    self.check_untightened_bounds_constraint(upper, m.x2, m.d3, m.disjunction, {m.d1: 3, m.d2: 10}, upper=1)