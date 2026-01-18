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
def check_disj_constraint(self, c1, upper, auxVar1, auxVar2):
    self.assertIsNone(c1.lower)
    self.assertEqual(value(c1.upper), upper)
    repn = generate_standard_repn(c1.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertEqual(repn.constant, 0)
    self.assertIs(repn.linear_vars[0], auxVar1)
    self.assertIs(repn.linear_vars[1], auxVar2)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertEqual(repn.linear_coefs[1], 1)