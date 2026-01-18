import pyomo.common.unittest as unittest
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
def test_active_parent_disjunct_target(self):
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d1.sub1 = Disjunct()
    m.d1.sub2 = Disjunct()
    m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
    TransformationFactory('gdp.bigm').apply_to(m, targets=m.d1.disj)
    m.d1.indicator_var.fix(1)
    TransformationFactory('gdp.reclassify').apply_to(m)
    self.assertIs(m.d1.ctype, Block)
    self.assertIs(m.d1.sub1.ctype, Block)
    self.assertIs(m.d1.sub2.ctype, Block)