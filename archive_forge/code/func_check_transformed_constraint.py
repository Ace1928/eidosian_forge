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
def check_transformed_constraint(self, cons, dis, lb, ind_var):
    hull = TransformationFactory('gdp.hull')
    self.assertEqual(len(cons), 1)
    cons = cons[0]
    self.assertTrue(cons.active)
    self.assertIsNone(cons.lower)
    self.assertEqual(value(cons.upper), 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, dis, -1)
    ct.check_linear_coef(self, repn, ind_var, lb)
    orig = ind_var.parent_block().c
    self.assertIs(hull.get_src_constraint(cons), orig)