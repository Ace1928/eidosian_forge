import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_issue_2819(self):
    m = ConcreteModel()
    m.x = Var()
    m.z = Var()
    t = 0.55
    m.x.fix(3.5)
    e = (m.x - 4) ** 2 + (m.z - 1) ** 2 - t
    tc = StorageTreeChecker(m)
    smap = SymbolMap()
    test = expression_to_string(e, tc, smap=smap)
    self.assertEqual(test, ('power((3.5 + (-4)), 2) + power((z + (-1)), 2) + (-0.55)', False))