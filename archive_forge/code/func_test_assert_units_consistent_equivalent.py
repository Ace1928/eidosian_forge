import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
def test_assert_units_consistent_equivalent(self):
    u = units
    m = ConcreteModel()
    m.dx = Var(units=u.m, initialize=0.10188943773836046)
    m.dy = Var(units=u.m, initialize=0.0)
    m.vx = Var(units=u.m / u.s, initialize=0.7071067769802851)
    m.vy = Var(units=u.m / u.s, initialize=0.7071067769802851)
    m.t = Var(units=u.min, bounds=(1e-05, 10.0), initialize=0.0024015570927624456)
    m.theta = Var(bounds=(0, 0.49 * 3.14), initialize=0.7853981693583533, units=u.radians)
    m.a = Param(initialize=-32.2, units=u.ft / u.s ** 2)
    m.x_unitless = Var()
    m.obj = Objective(expr=m.dx, sense=maximize)
    m.vx_con = Constraint(expr=m.vx == 1.0 * u.m / u.s * cos(m.theta))
    m.vy_con = Constraint(expr=m.vy == 1.0 * u.m / u.s * sin(m.theta))
    m.dx_con = Constraint(expr=m.dx == m.vx * u.convert(m.t, to_units=u.s))
    m.dy_con = Constraint(expr=m.dy == m.vy * u.convert(m.t, to_units=u.s) + 0.5 * u.convert(m.a, to_units=u.m / u.s ** 2) * u.convert(m.t, to_units=u.s) ** 2)
    m.ground = Constraint(expr=m.dy == 0)
    m.unitless_con = Constraint(expr=m.x_unitless == 5.0)
    assert_units_consistent(m)
    assert_units_consistent(m.dx)
    assert_units_consistent(m.x_unitless)
    assert_units_consistent(m.vx_con)
    assert_units_consistent(m.unitless_con)
    assert_units_equivalent(m.dx, m.dy)
    assert_units_equivalent(m.x_unitless, u.dimensionless)
    assert_units_equivalent(m.x_unitless, None)
    assert_units_equivalent(m.vx_con.body, u.m / u.s)
    assert_units_equivalent(m.unitless_con.body, u.dimensionless)
    assert_units_equivalent(m.dx, m.dy)
    assert_units_equivalent(m.x_unitless, u.dimensionless)
    assert_units_equivalent(m.x_unitless, None)
    assert_units_equivalent(m.vx_con.body, u.m / u.s)
    m.broken = Constraint(expr=m.dy == 42.0 * u.kg)
    with self.assertRaises(UnitsError):
        assert_units_consistent(m)
    assert_units_consistent(m.dx)
    assert_units_consistent(m.vx_con)
    with self.assertRaises(UnitsError):
        assert_units_consistent(m.broken)
    self.assertTrue(check_units_equivalent(m.dx, m.dy))
    self.assertFalse(check_units_equivalent(m.dx, m.vx))