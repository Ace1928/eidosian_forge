import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_negative_float_double_operator(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var(bounds=(0, 6))
    m.c = Constraint(expr=m.x * m.y * -2 == 0)
    m.c2 = Constraint(expr=m.z ** (-1.5) == 0)
    m.o = Objective(expr=m.z)
    m.y.fix(-7)
    m.x.fix(4)
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(expression_to_string(m.c.body, tc, smap=smap), ('4*(-7)*(-2)', False))
    self.assertEqual(expression_to_string(m.c2.body, tc, smap=smap), ('x1 ** (-1.5)', False))