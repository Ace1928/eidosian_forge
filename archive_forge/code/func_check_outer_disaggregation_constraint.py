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
def check_outer_disaggregation_constraint(self, cons, var, disj1, disj2, rhs=None):
    if rhs is None:
        rhs = var
    hull = TransformationFactory('gdp.hull')
    self.assertTrue(cons.active)
    self.assertEqual(cons.lower, 0)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    ct.check_linear_coef(self, repn, rhs, 1)
    ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj1), -1)
    ct.check_linear_coef(self, repn, hull.get_disaggregated_var(var, disj2), -1)