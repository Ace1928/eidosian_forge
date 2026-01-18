import math
from chempy.chemistry import Equilibrium
from chempy.util._expr import Expr
from chempy.util.testing import requires
from chempy.units import (
from ..expressions import MassActionEq, GibbsEqConst
@requires(units_library)
def test_GibbsEqConst__units():
    R, T = (dc.molar_gas_constant, 298.15 * du.K)
    DH = -4000.0 * du.J / du.mol
    DS = 16 * du.J / du.K / du.mol
    be = Backend()
    gee = GibbsEqConst([DH / R, DS / R])
    res = gee.eq_const({'temperature': T}, backend=be)
    ref = be.exp(-(DH - T * DS) / (R * T))
    assert allclose(res, ref)