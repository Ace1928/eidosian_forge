from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def check_bounds_constraint_ub(self, constraint, ub, dis_var, ind_var):
    hull = TransformationFactory('gdp.hull')
    self.assertIsInstance(constraint, Constraint)
    self.assertTrue(constraint.active)
    self.assertEqual(len(constraint), 1)
    self.assertTrue(constraint['ub'].active)
    self.assertEqual(constraint['ub'].upper, 0)
    self.assertIsNone(constraint['ub'].lower)
    repn = generate_standard_repn(constraint['ub'].body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, dis_var, 1)
    ct.check_linear_coef(self, repn, ind_var, -ub)
    self.assertIs(constraint, hull.get_var_bounds_constraint(dis_var))