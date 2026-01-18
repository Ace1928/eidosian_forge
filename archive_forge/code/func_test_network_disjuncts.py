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
@unittest.skipUnless(ct.linear_solvers, 'Could not find a linear solver')
def test_network_disjuncts(self):
    ct.check_network_disjuncts(self, True, 'between_steps', num_partitions=2)
    ct.check_network_disjuncts(self, False, 'between_steps', num_partitions=2)