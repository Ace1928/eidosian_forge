import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires(units_library, 'numpy')
@pytest.mark.parametrize('R_from_constants', [False, True])
def test_ArrheniusMassAction__units(R_from_constants):
    import numpy as np
    Ea = 40000.0 * u.J / u.mol
    R = default_constants.molar_gas_constant if R_from_constants else 8.3145 * u.J / u.mol / u.K
    A, Ea_over_R = (120000000000.0 / u.molar ** 2 / u.second, Ea / R)
    ref1 = A * np.exp(-to_unitless(Ea_over_R / (290 * u.K)))
    arrh = Arrhenius([A, Ea_over_R])
    assert allclose(arrh({'temperature': 290 * u.K}), ref1)
    ama = MassAction(arrh)
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, ama, {'B': 1})
    T_ = 'temperature'

    def ref(v):
        return 120000000000.0 / u.molar ** 2 / u.second * math.exp(-Ea_over_R.simplified / v[T_]) * v['B'] * v['A'] ** 2
    ma = r.rate_expr()
    for params in [(11.0 * u.molar, 13.0 * u.molar, 17.0 * u.molar, 311.2 * u.kelvin), (12 * u.molar, 8 * u.molar, 5 * u.molar, 270 * u.kelvin)]:
        var = dict(zip(['A', 'B', 'C', T_], params))
        ref_val = ref(var)
        assert abs((ma(var, reaction=r) - ref_val) / ref_val) < 1e-14