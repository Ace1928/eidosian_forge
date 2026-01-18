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
def check_inner_xor_constraint(self, inner_disjunction, outer_disjunct, bigm):
    inner_xor = inner_disjunction.algebraic_constraint
    sum_indicators = sum((d.binary_indicator_var for d in inner_disjunction.disjuncts))
    assertExpressionsEqual(self, inner_xor.expr, sum_indicators == 1)
    self.assertFalse(inner_xor.active)
    cons = bigm.get_transformed_constraints(inner_xor)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ct.check_obj_in_active_tree(self, lb)
    lb_expr = self.simplify_cons(lb, leq=False)
    assertExpressionsEqual(self, lb_expr, 1.0 <= sum_indicators - outer_disjunct.binary_indicator_var + 1.0)
    ub = cons[1]
    ct.check_obj_in_active_tree(self, ub)
    ub_expr = self.simplify_cons(ub, leq=True)
    assertExpressionsEqual(self, ub_expr, sum_indicators + outer_disjunct.binary_indicator_var - 1 <= 1.0)