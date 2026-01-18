import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
def test_disjunction_not_sum_to_1(self):
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.d = Disjunction(expr=[m.d1, m.d2], xor=False)
    m.d1.indicator_var.set_value(False)
    m.d2.indicator_var.set_value(False)
    with self.assertRaises(GDP_Error):
        TransformationFactory('gdp.fix_disjuncts').apply_to(m)