import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_fixed_var_to_string(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.z.fix(-3)
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(expression_to_string(m.x + m.y - m.z, tc, smap=smap), ('x1 + x2 + 3', False))
    m.z.fix(-400)
    self.assertEqual(expression_to_string(m.z + m.y - m.z, tc, smap=smap), ('(-400) + x2 + 400', False))
    m.z.fix(8.8)
    self.assertEqual(expression_to_string(m.x + m.z - m.y, tc, smap=smap), ('x1 + 8.8 - x2', False))
    m.z.fix(-8.8)
    self.assertEqual(expression_to_string(m.x * m.z - m.y, tc, smap=smap), ('x1*(-8.8) - x2', False))