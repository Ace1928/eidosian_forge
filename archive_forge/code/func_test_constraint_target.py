from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_constraint_target(self):
    m = self.make_model()
    TransformationFactory('contrib.logical_to_disjunctive').apply_to(m, targets=[m.block.c1])
    transBlock = m.block._logical_to_disjunctive
    self.assertEqual(len(transBlock.auxiliary_vars), 3)
    self.assertEqual(len(transBlock.transformed_constraints), 4)
    self.assertEqual(len(transBlock.auxiliary_disjuncts), 0)
    self.assertEqual(len(transBlock.auxiliary_disjunctions), 0)
    self.check_block_c1_transformed(m, transBlock)
    self.assertTrue(m.block.c2.active)
    self.assertTrue(m.c1.active)