import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
def test_new_block_created(self):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.binary_multiplication').apply_to(m)
    transBlock = m.component('_pyomo_gdp_binary_multiplication_reformulation')
    self.assertIsInstance(transBlock, Block)
    disjBlock = transBlock.component('relaxedDisjuncts')
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    self.assertIs(m.d[0].transformation_block, disjBlock[0])
    self.assertIs(m.d[1].transformation_block, disjBlock[1])