from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, units_library)
def test_ReactionSystem__from_string():
    rs = ReactionSystem.from_string('-> H + OH; Radiolytic(2.1e-7)', checks=())
    assert rs.rxns[0].reac == {}
    assert rs.rxns[0].prod == {'H': 1, 'OH': 1}
    assert rs.rxns[0].param.args == [2.1e-07]
    ref = 2.1e-07 * 0.15 * 998
    assert rs.rates({'doserate': 0.15, 'density': 998}) == {'H': ref, 'OH': ref}
    r2, = ReactionSystem.from_string('H2O + H2O + H+ -> H3O+ + H2O').rxns
    assert r2.reac == {'H2O': 2, 'H+': 1}
    assert r2.prod == {'H2O': 1, 'H3O+': 1}
    rs2 = ReactionSystem.from_string("\n #  H2O -> OH + H\n  H+ + OH- -> H2O; 4*pi*(4e-9*m**2/s + 2e-9*m**2/s)*0.44*nm*Avogadro_constant; ref='made up #hashtag'  # comment\n#H+ + OH- -> H2O\n")
    assert len(rs2.rxns) == 1
    assert sorted(rs2.substances.keys()) == sorted('H2O H+ OH-'.split())
    assert allclose(rs2.rxns[0].param, 19978600000.0 / default_units.M / default_units.s, rtol=1e-05)
    assert rs2.rxns[0].ref == 'made up #hashtag'