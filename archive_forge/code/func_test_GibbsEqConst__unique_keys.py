import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
def test_GibbsEqConst__unique_keys():
    R, T = (8.314, 298.15)
    dH, dS = (-4000.0, 16)
    gee = GibbsEqConst(unique_keys=('dH1', 'dS1'))
    ref = math.exp(-(dH - T * dS) / (R * T))
    assert abs((gee.eq_const({'temperature': T, 'dH1': dH / R, 'dS1': dS / R}) - ref) / ref) < 1e-14