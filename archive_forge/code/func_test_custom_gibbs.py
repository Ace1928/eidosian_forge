import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
def test_custom_gibbs():
    R, T = (8.314, 298.15)
    dH, dS = (-4000.0, 16)
    MyGibbs = MassActionEq.from_callback(_gibbs, parameter_keys=('temperature', 'R'), argument_names=('H', 'S', 'Cp', 'Tref'))
    dCp = 123.45
    Tref = 242
    gee2 = MyGibbs([dH, dS, dCp, Tref])
    dH2 = dH + dCp * (T - Tref)
    dS2 = dS + dCp * math.log(T / Tref)
    ref2 = math.exp(-(dH2 - T * dS2) / (R * T))
    assert abs((gee2.eq_const({'temperature': T, 'R': R}) - ref2) / ref2) < 1e-14