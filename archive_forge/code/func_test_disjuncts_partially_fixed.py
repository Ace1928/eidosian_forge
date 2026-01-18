import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
def test_disjuncts_partially_fixed(self):
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.d = Disjunction(expr=[m.d1, m.d2])
    m.another1 = Disjunct()
    m.another2 = Disjunct()
    m.another = Disjunction(expr=[m.another1, m.another2])
    m.d1.indicator_var.set_value(True)
    m.d2.indicator_var.set_value(False)
    with self.assertRaisesRegex(GDP_Error, "The value of the indicator_var of Disjunct 'another1' is None. All indicator_vars must have values before calling 'fix_disjuncts'."):
        TransformationFactory('gdp.fix_disjuncts').apply_to(m)