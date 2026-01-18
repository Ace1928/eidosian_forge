from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def check_and_constraints(self, a, b1, z, transBlock):
    assertExpressionsEqual(self, transBlock.transformed_constraints[1].expr, z <= a)
    assertExpressionsEqual(self, transBlock.transformed_constraints[2].expr, z <= b1)
    assertExpressionsEqual(self, transBlock.transformed_constraints[3].expr, 1 - z <= 2 - (a + b1))
    assertExpressionsEqual(self, transBlock.transformed_constraints[4].expr, z >= 1)