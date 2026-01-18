import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def test_ArrheniusMassAction():
    A, Ea_over_R = (120000000000.0, 40000.0 / 8.3145)
    ama = MassAction(Arrhenius([A, Ea_over_R]))
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, ama, {'B': 1})
    T_ = 'temperature'

    def ref(v):
        return 120000000000.0 * math.exp(-Ea_over_R / v[T_]) * v['B'] * v['A'] ** 2
    ma = r.rate_expr()
    for params in [(11.0, 13.0, 17.0, 311.2), (12, 8, 5, 270)]:
        var = dict(zip(['A', 'B', 'C', T_], params))
        ref_var = ref(var)
        assert abs((ma(var, reaction=r) - ref_var) / ref_var) < 1e-14
    with pytest.raises(ValueError):
        Arrhenius([A, Ea_over_R, 1, A])