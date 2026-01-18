import math
import pytest
from chempy import Reaction, ReactionSystem, Substance
from chempy.chemistry import Equilibrium
from chempy.units import (
from chempy.util._expr import Expr
from chempy.util.parsing import parsing_library
from chempy.util.testing import requires
from ..rates import RateExpr, MassAction, Arrhenius, Radiolytic, mk_Radiolytic, Eyring
def test_mk_Radiolytic():
    R1 = mk_Radiolytic()
    R2 = mk_Radiolytic()
    assert R1 is R2
    RABG = mk_Radiolytic('alpha', 'beta', 'gamma')
    rxn = Reaction({}, {'H': 2}, RABG([3, 5, 7], 'ya yb yg'.split()))
    rat = rxn.rate({'doserate_alpha': 11, 'doserate_beta': 13, 'doserate_gamma': 17, 'density': 0.7})
    assert abs(rat['H'] - 0.7 * 2 * (3 * 11 + 5 * 13 + 7 * 17)) < 1e-13
    assert RABG.parameter_keys == ('density', 'doserate_alpha', 'doserate_beta', 'doserate_gamma')
    assert RABG.argument_names == tuple(('radiolytic_yield_%s' % k for k in 'alpha beta gamma'.split()))
    assert rxn.param.unique_keys == ('ya', 'yb', 'yg')
    rat2 = rxn.rate({'doserate_alpha': 11, 'doserate_beta': 13, 'doserate_gamma': 17, 'density': 0.7, 'ya': 23, 'yb': 29, 'yg': 31})
    assert abs(rat2['H'] - 0.7 * 2 * (23 * 11 + 29 * 13 + 31 * 17)) < 1e-13