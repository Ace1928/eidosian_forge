import pyomo.common.unittest as unittest
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
def test_active_parent_block(self):
    m = ConcreteModel()
    m.d1 = Block()
    m.d1.sub1 = Disjunct()
    m.d1.sub2 = Disjunct()
    m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
    with self.assertRaises(GDP_Error):
        TransformationFactory('gdp.reclassify').apply_to(m)