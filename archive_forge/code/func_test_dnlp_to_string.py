import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_dnlp_to_string(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    lbl = NumericLabeler('x')
    smap = SymbolMap(lbl)
    tc = StorageTreeChecker(m)
    self.assertEqual(expression_to_string(ceil(m.x), tc, smap=smap), ('ceil(x1)', True))
    self.assertEqual(expression_to_string(floor(m.x), tc, smap=smap), ('floor(x1)', True))
    self.assertEqual(expression_to_string(abs(m.x), tc, smap=smap), ('abs(x1)', True))