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
@requires(parsing_library)
def test_Reaction__from_string():
    r = Reaction.from_string('H2O -> H+ + OH-; 1e-4', 'H2O H+ OH-'.split())
    assert r.reac == {'H2O': 1} and r.prod == {'H+': 1, 'OH-': 1}
    with pytest.raises(ValueError):
        Reaction.from_string('H2O -> H+ + OH-; 1e-4', 'H2O H OH-'.split())
    r2 = Reaction.from_string("H2O -> H+ + OH-; 1e-4; ref='important_paper'")
    assert r2.ref == 'important_paper'
    with pytest.raises(ValueError):
        Reaction.from_string('H2O -> H2O')
    Reaction.from_string('H2O -> H2O; None; checks=()')
    with pytest.raises(ValueError):
        Reaction({'H2O': 2}, {'H2O2': 2, 'O2': -2})
    r4 = Reaction({'H+': 2, 'OH-': 1}, {'H2O': 2}, 42.0)
    assert Reaction.from_string(str(r4), 'H+ OH- H2O') == r4
    assert Reaction.from_string(str(r4), None) == r4
    r5 = Reaction.from_string('H2O2 -> 0.5 O2 + H2O', checks=[c for c in Reaction.default_checks if c != 'all_integral'])
    r6 = r5.copy()
    assert r5 == r6
    r7 = Reaction.from_string("H2O -> H + OH; None; data=dict(ref='foo; bar; baz;')  # foobar")
    assert r7.data['ref'] == 'foo; bar; baz;'