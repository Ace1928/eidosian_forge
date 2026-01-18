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
def test_locally_declared_var_bounds_used_globally(self):
    m = models.localVar()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    y_disagg = m.disj2.transformation_block.disaggregatedVars.component('disj2.y')
    cons = hull.get_var_bounds_constraint(y_disagg)
    lb = cons['lb']
    self.assertIsNone(lb.lower)
    self.assertEqual(value(lb.upper), 0)
    repn = generate_standard_repn(lb.body)
    self.assertTrue(repn.is_linear())
    ct.check_linear_coef(self, repn, m.disj2.indicator_var, 1)
    ct.check_linear_coef(self, repn, y_disagg, -1)
    ub = cons['ub']
    self.assertIsNone(ub.lower)
    self.assertEqual(value(ub.upper), 0)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    ct.check_linear_coef(self, repn, y_disagg, 1)
    ct.check_linear_coef(self, repn, m.disj2.indicator_var, -3)