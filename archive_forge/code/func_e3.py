import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.network import Port, Arc
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.base.units_container import pint_available, UnitsError
from pyomo.util.check_units import (
@m.Expression(m.S)
def e3(blk, i):
    if i == 1:
        return m.t + 10 * u.m
    return m.t + 10 * u.s