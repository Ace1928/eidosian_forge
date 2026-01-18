import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires(units_library)
def test_Radiolytic__units():

    def _check(r):
        res = r({'doserate': 0.15 * u.gray / u.second, 'density': 0.998 * u.kg / u.decimetre ** 3})
        ref = 0.15 * 0.998 * 2.1e-07 * u.molar / u.second
        assert abs(to_unitless((res - ref) / ref)) < 1e-15
    _check(Radiolytic([2.1e-07 * u.mol / u.joule]))
    _check(Radiolytic([2.0261921896167396 * u.per100eV]))