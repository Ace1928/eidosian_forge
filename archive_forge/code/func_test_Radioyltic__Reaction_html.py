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
def test_Radioyltic__Reaction_html():
    rate = Radiolytic([2.1 * u.per100eV])
    rxn = Reaction({}, {'H': 1}, rate)
    H = Substance.from_formula('H')
    html = rxn.html({'H': H}, with_param=True)
    assert html == ' &rarr; H&#59; %s' % str(rate)