import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_gams_expanded_arcs(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.CON1 = Port()
    m.CON1.add(m.x, 'v')
    m.CON2 = Port()
    m.CON2.add(m.y, 'v')
    m.c = Arc(source=m.CON1, destination=m.CON2)
    TransformationFactory('network.expand_arcs').apply_to(m)
    m.o = Objective(expr=m.x)
    outs = StringIO()
    io_options = dict(symbolic_solver_labels=True)
    m.write(outs, format='gams', io_options=io_options)
    self.assertIn('x - y', outs.getvalue())