import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def extensive_recycle_model(self):

    def build_in_out(b):
        b.flow_in = Var(m.comps)
        b.mass_in = Var()
        b.temperature_in = Var()
        b.pressure_in = Var()
        b.expr_var_idx_in = Var(m.comps)

        @b.Expression(m.comps)
        def expr_idx_in(b, i):
            return -b.expr_var_idx_in[i]
        b.expr_var_in = Var()
        b.expr_in = -b.expr_var_in
        b.flow_out = Var(m.comps)
        b.mass_out = Var()
        b.temperature_out = Var()
        b.pressure_out = Var()
        b.expr_var_idx_out = Var(m.comps)

        @b.Expression(m.comps)
        def expr_idx_out(b, i):
            return -b.expr_var_idx_out[i]
        b.expr_var_out = Var()
        b.expr_out = -b.expr_var_out
        b.inlet = Port(rule=inlet)
        b.outlet = Port(rule=outlet)
        b.initialize = MethodType(initialize, b)

    def inlet(b):
        return dict(flow=(b.flow_in, Port.Extensive), mass=(b.mass_in, Port.Extensive), temperature=b.temperature_in, pressure=b.pressure_in, expr_idx=(b.expr_idx_in, Port.Extensive), expr=(b.expr_in, Port.Extensive))

    def outlet(b):
        return dict(flow=(b.flow_out, Port.Extensive), mass=(b.mass_out, Port.Extensive), temperature=b.temperature_out, pressure=b.pressure_out, expr_idx=(b.expr_idx_out, Port.Extensive), expr=(b.expr_out, Port.Extensive))

    def initialize(self):
        for i in self.flow_out:
            self.flow_out[i].value = value(self.flow_in[i])
        self.mass_out.value = value(self.mass_in)
        for i in self.expr_var_idx_out:
            self.expr_var_idx_out[i].value = value(self.expr_var_idx_in[i])
        self.expr_var_out.value = value(self.expr_var_in)
        self.temperature_out.value = value(self.temperature_in)
        self.pressure_out.value = value(self.pressure_in)

    def nop(self):
        pass
    m = ConcreteModel()
    m.comps = Set(initialize=['A', 'B', 'C'])
    m.feed = Block()
    m.feed.flow_out = Var(m.comps)
    m.feed.mass_out = Var()
    m.feed.temperature_out = Var()
    m.feed.pressure_out = Var()
    m.feed.expr_var_idx_out = Var(m.comps)

    @m.feed.Expression(m.comps)
    def expr_idx_out(b, i):
        return -b.expr_var_idx_out[i]
    m.feed.expr_var_out = Var()
    m.feed.expr_out = -m.feed.expr_var_out
    m.feed.outlet = Port(rule=outlet)
    m.feed.initialize = MethodType(nop, m.feed)
    m.mixer = Block()
    build_in_out(m.mixer)
    m.unit = Block()
    build_in_out(m.unit)
    m.splitter = Block()
    build_in_out(m.splitter)
    m.prod = Block()
    m.prod.flow_in = Var(m.comps)
    m.prod.mass_in = Var()
    m.prod.temperature_in = Var()
    m.prod.pressure_in = Var()
    m.prod.actual_var_idx_in = Var(m.comps)
    m.prod.actual_var_in = Var()

    @m.prod.Port()
    def inlet(b):
        return dict(flow=(b.flow_in, Port.Extensive), mass=(b.mass_in, Port.Extensive), temperature=b.temperature_in, pressure=b.pressure_in, expr_idx=(b.actual_var_idx_in, Port.Extensive), expr=(b.actual_var_in, Port.Extensive))
    m.prod.initialize = MethodType(nop, m.prod)

    @m.Arc(directed=True)
    def stream_feed_to_mixer(m):
        return (m.feed.outlet, m.mixer.inlet)

    @m.Arc(directed=True)
    def stream_mixer_to_unit(m):
        return (m.mixer.outlet, m.unit.inlet)

    @m.Arc(directed=True)
    def stream_unit_to_splitter(m):
        return (m.unit.outlet, m.splitter.inlet)

    @m.Arc(directed=True)
    def stream_splitter_to_mixer(m):
        return (m.splitter.outlet, m.mixer.inlet)

    @m.Arc(directed=True)
    def stream_splitter_to_prod(m):
        return (m.splitter.outlet, m.prod.inlet)
    rec = 0.1
    prod = 1 - rec
    m.splitter.outlet.set_split_fraction(m.stream_splitter_to_mixer, rec)
    m.splitter.outlet.set_split_fraction(m.stream_splitter_to_prod, prod)
    TransformationFactory('network.expand_arcs').apply_to(m)
    m.feed.flow_out['A'].fix(100)
    m.feed.flow_out['B'].fix(200)
    m.feed.flow_out['C'].fix(300)
    m.feed.mass_out.fix(400)
    m.feed.expr_var_idx_out['A'].fix(10)
    m.feed.expr_var_idx_out['B'].fix(20)
    m.feed.expr_var_idx_out['C'].fix(30)
    m.feed.expr_var_out.fix(40)
    m.feed.temperature_out.fix(450)
    m.feed.pressure_out.fix(128)
    return m