from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_block_target(self):
    m = self.make_model()
    TransformationFactory('contrib.logical_to_disjunctive').apply_to(m, targets=[m.block])
    self.check_block_transformed(m)
    self.assertTrue(m.c1.active)