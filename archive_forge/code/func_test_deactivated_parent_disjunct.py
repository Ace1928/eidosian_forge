import pyomo.common.unittest as unittest
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
def test_deactivated_parent_disjunct(self):
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d1.sub1 = Disjunct()
    m.d1.sub2 = Disjunct()
    m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
    m.d1.deactivate()
    TransformationFactory('gdp.reclassify').apply_to(m)
    self.assertIs(m.d1.ctype, Block)
    self.assertIs(m.d1.sub1.ctype, Block)
    self.assertIs(m.d1.sub2.ctype, Block)