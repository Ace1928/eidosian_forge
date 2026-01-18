from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeNetworkDisjunction(minimize=True):
    """creates a GDP model with pyomo.network components"""
    m = ConcreteModel()
    m.feed = feed = Block()
    m.wkbx = wkbx = Block()
    m.dest = dest = Block()
    m.orange = orange = Disjunct()
    m.blue = blue = Disjunct()
    m.orange_or_blue = Disjunction(expr=[orange, blue])
    blue.blue_box = blue_box = Block()
    feed.x = Var(bounds=(0, 1))
    wkbx.x = Var(bounds=(0, 1))
    dest.x = Var(bounds=(0, 1))
    wkbx.inlet = ntwk.Port(initialize={'x': wkbx.x})
    wkbx.outlet = ntwk.Port(initialize={'x': wkbx.x})
    feed.outlet = ntwk.Port(initialize={'x': feed.x})
    dest.inlet = ntwk.Port(initialize={'x': dest.x})
    blue_box.x = Var(bounds=(0, 1))
    blue_box.x_wkbx = Var(bounds=(0, 1))
    blue_box.x_dest = Var(bounds=(0, 1))
    blue_box.inlet_feed = ntwk.Port(initialize={'x': blue_box.x})
    blue_box.outlet_wkbx = ntwk.Port(initialize={'x': blue_box.x})
    blue_box.inlet_wkbx = ntwk.Port(initialize={'x': blue_box.x_wkbx})
    blue_box.outlet_dest = ntwk.Port(initialize={'x': blue_box.x_dest})
    blue_box.multiplier_constr = Constraint(expr=blue_box.x_dest == 2 * blue_box.x_wkbx)
    orange.a1 = ntwk.Arc(source=feed.outlet, destination=wkbx.inlet)
    orange.a2 = ntwk.Arc(source=wkbx.outlet, destination=dest.inlet)
    blue.a1 = ntwk.Arc(source=feed.outlet, destination=blue_box.inlet_feed)
    blue.a2 = ntwk.Arc(source=blue_box.outlet_wkbx, destination=wkbx.inlet)
    blue.a3 = ntwk.Arc(source=wkbx.outlet, destination=blue_box.inlet_wkbx)
    blue.a4 = ntwk.Arc(source=blue_box.outlet_dest, destination=dest.inlet)
    if minimize:
        m.obj = Objective(expr=m.dest.x)
    else:
        m.obj = Objective(expr=m.dest.x, sense=maximize)
    feed.x.fix(0.42)
    return m