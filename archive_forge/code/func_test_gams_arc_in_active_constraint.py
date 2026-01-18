import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_gams_arc_in_active_constraint(self):
    m = ConcreteModel()
    m.b1 = Block()
    m.b2 = Block()
    m.b1.x = Var()
    m.b2.x = Var()
    m.b1.c = Port()
    m.b1.c.add(m.b1.x)
    m.b2.c = Port()
    m.b2.c.add(m.b2.x)
    m.c = Arc(source=m.b1.c, destination=m.b2.c)
    m.o = Objective(expr=m.b1.x)
    outs = StringIO()
    with self.assertRaises(RuntimeError):
        m.write(outs, format='gams')