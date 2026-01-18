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
def check_aux_var_bounds(self, aux_vars1, aux_vars2, aux11lb, aux11ub, aux12lb, aux12ub, aux21lb, aux21ub, aux22lb, aux22ub):
    self.assertEqual(len(aux_vars1), 2)
    self.assertAlmostEqual(aux_vars1[0].lb, aux11lb, places=6)
    self.assertAlmostEqual(aux_vars1[0].ub, aux11ub, places=6)
    self.assertAlmostEqual(aux_vars1[1].lb, aux12lb, places=6)
    self.assertAlmostEqual(aux_vars1[1].ub, aux12ub, places=6)
    self.assertAlmostEqual(len(aux_vars2), 2)
    self.assertAlmostEqual(aux_vars2[0].lb, aux21lb, places=6)
    self.assertAlmostEqual(aux_vars2[0].ub, aux21ub, places=6)
    self.assertAlmostEqual(aux_vars2[1].lb, aux22lb, places=6)
    self.assertAlmostEqual(aux_vars2[1].ub, aux22ub, places=6)