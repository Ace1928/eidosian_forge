from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires(parsing_library, units_library)
def test_Reaction_from_string__units():
    r5 = Reaction.from_string('2 H2O2 -> O2 + 2 H2O; 1e-7/molar/second', 'H2O O2 H2O2')
    assert to_unitless(r5.param, 1 / default_units.molar / default_units.second) == 1e-07
    r6 = Reaction.from_string('->', checks=())
    assert r6.reac == {} and r6.prod == {}
    r7 = Reaction.from_string('2 A -> B; exp(log(2e-3))*metre**3/mol/hour', None)
    assert r7.reac == {'A': 2} and r7.prod == {'B': 1}
    assert allclose(r7.param, 0.002 * default_units.metre ** 3 / default_units.mol / default_units.hour)
    with pytest.raises(ValueError):
        Reaction.from_string('2 A -> B; 2e-3/hour', None)
    r8 = Reaction.from_string('A -> B; "k"')
    assert r8.rate_expr().args is None
    assert r8.rate_expr().unique_keys == ('k',)
    r9 = Reaction.from_string('A -> B; 42.0')
    assert r9.rate_expr().args == [42.0]
    assert r9.rate_expr().unique_keys is None
    Reaction.from_string('H+ + OH- -> H2O; 1e10/M/s', 'H2O H+ OH-'.split())
    with pytest.raises(ValueError):
        Reaction.from_string('H2O -> H+ + OH-; 1e-4/M/s', 'H2O H+ OH-'.split())