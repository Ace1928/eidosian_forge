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
def check_first_disjunct_constraint(self, disj1c, x, ind_var):
    self.assertEqual(len(disj1c), 1)
    cons = disj1c[0]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 1)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_quadratic())
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertEqual(len(repn.quadratic_vars), 4)
    ct.check_linear_coef(self, repn, ind_var, 143)
    self.assertEqual(repn.constant, -143)
    for i in range(1, 5):
        ct.check_squared_term_coef(self, repn, x[i], 1)