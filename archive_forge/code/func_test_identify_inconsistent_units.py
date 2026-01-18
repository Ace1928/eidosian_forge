import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
def test_identify_inconsistent_units(self):
    u = units
    m = ConcreteModel()
    m.S = Set(initialize=[1, 2, 3])
    m.x = Var(units=u.m)
    m.t = Var(units=u.s)
    m.c1 = Constraint(expr=m.x == 10 * u.m)
    m.c2 = Constraint(expr=m.x == m.t)

    @m.Constraint(m.S)
    def c3(blk, i):
        if i == 1:
            return m.t == 10 * u.m
        return m.t == 10 * u.s
    m.e1 = Expression(expr=m.x + 10 * u.m)
    m.e2 = Expression(expr=m.x + m.t)

    @m.Expression(m.S)
    def e3(blk, i):
        if i == 1:
            return m.t + 10 * u.m
        return m.t + 10 * u.s
    m.o1 = Objective(expr=m.x + 10 * u.m)
    m.o2 = Objective(expr=m.x + m.t)

    @m.Objective(m.S)
    def o3(blk, i):
        if i == 1:
            return m.t + 10 * u.m
        return m.t + 10 * u.s
    failures = identify_inconsistent_units(m)
    assert len(failures) == 6
    assert m.c2 in failures
    assert m.c3[1] in failures
    assert m.e2 in failures
    assert m.e3[1] in failures
    assert m.o2 in failures
    assert m.o3[1] in failures