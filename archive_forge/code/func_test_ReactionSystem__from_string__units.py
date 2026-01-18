from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, units_library)
def test_ReactionSystem__from_string__units():
    r3, = ReactionSystem.from_string('(H2O) -> e-(aq) + H+ + OH; Radiolytic(2.1e-7*mol/J)').rxns
    assert len(r3.reac) == 0 and r3.inact_reac == {'H2O': 1}
    assert r3.prod == {'e-(aq)': 1, 'H+': 1, 'OH': 1}
    from chempy.kinetics.rates import Radiolytic
    mol, J = (default_units.mol, default_units.J)
    assert r3.param == Radiolytic(2.1e-07 * mol / J)
    assert r3.param != Radiolytic(2e-07 * mol / J)
    assert r3.param != Radiolytic(2.1e-07)
    assert r3.order() == 0
    k = 0.0001 / default_units.second
    rs = ReactionSystem.from_string('\n    H2O -> H+ + OH-; {}\n    '.format(repr(k)))
    assert allclose(rs.rxns[0].param, k)