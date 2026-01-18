import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
@requires(parsing_library)
def test_Radiolytic__parsing():
    rxn = Reaction.from_string("-> H + OH; Radiolytic({'radiolytic_yield': 2.1e-7})", None)
    res = rxn.rate({'doserate': 0.15, 'density': 0.998})
    ref = 0.15 * 0.998 * 2.1e-07
    assert abs((res['H'] - ref) / ref) < 1e-15
    assert abs((res['OH'] - ref) / ref) < 1e-15
    gval, = rxn.rate_expr().g_values({}).values()
    assert abs(gval - 2.1e-07) < 1e-15