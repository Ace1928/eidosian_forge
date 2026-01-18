from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def checkMs(self, model, c11lb, c12lb, c21lb, c21ub, c22lb, c22ub):
    bigm = TransformationFactory('gdp.bigm')
    c = bigm.get_transformed_constraints(model.disjunct[0].c[1])
    self.assertEqual(len(c), 1)
    lb = c[0]
    repn = generate_standard_repn(lb.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, -c11lb)
    ct.check_linear_coef(self, repn, model.disjunct[0].indicator_var, c11lb)
    c = bigm.get_transformed_constraints(model.disjunct[0].c[2])
    self.assertEqual(len(c), 1)
    lb = c[0]
    repn = generate_standard_repn(lb.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, -c12lb)
    ct.check_linear_coef(self, repn, model.disjunct[0].indicator_var, c12lb)
    c = bigm.get_transformed_constraints(model.disjunct[1].c[1])
    self.assertEqual(len(c), 2)
    lb = c[0]
    ub = c[1]
    repn = generate_standard_repn(lb.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, -c21lb)
    ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21lb)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, -c21ub)
    ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21ub)
    c = bigm.get_transformed_constraints(model.disjunct[1].c[2])
    self.assertEqual(len(c), 2)
    lb = c[0]
    ub = c[1]
    repn = generate_standard_repn(lb.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, -c22lb)
    ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22lb)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, -c22ub)
    ct.check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22ub)