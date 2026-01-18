import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
def check_second_disjunction_global_constraint_disj2(self, c, aux_vars2):
    m = c.model()
    self.assertEqual(len(c), 2)
    c1 = c[0]
    self.assertIsNone(c1.lower)
    self.assertEqual(c1.upper, 0)
    repn = generate_standard_repn(c1.body)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.linear_coefs[0], -4)
    self.assertIs(repn.linear_vars[0], m.x[1])
    self.assertEqual(repn.linear_coefs[1], -1)
    self.assertIs(repn.linear_vars[1], aux_vars2[0])
    self.assertEqual(len(repn.quadratic_vars), 1)
    self.assertIs(repn.quadratic_vars[0][0], m.x[1])
    self.assertIs(repn.quadratic_vars[0][1], m.x[1])
    self.assertEqual(repn.quadratic_coefs[0], 1)
    self.assertIsNone(repn.nonlinear_expr)
    c2 = c[1]
    self.assertIsNone(c2.lower)
    self.assertEqual(c2.upper, 0)
    repn = generate_standard_repn(c2.body)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.linear_coefs[0], -6)
    self.assertIs(repn.linear_vars[0], m.x[2])
    self.assertEqual(repn.linear_coefs[1], -1)
    self.assertIs(repn.linear_vars[1], aux_vars2[1])
    self.assertEqual(len(repn.quadratic_vars), 1)
    self.assertIs(repn.quadratic_vars[0][0], m.x[2])
    self.assertIs(repn.quadratic_vars[0][1], m.x[2])
    self.assertEqual(repn.quadratic_coefs[0], 1)
    self.assertIsNone(repn.nonlinear_expr)