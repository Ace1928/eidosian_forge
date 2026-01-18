from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library)
def test_ReactionSystem__add():
    rs1 = ReactionSystem.from_string('\n'.join(['2 H2O2 -> O2 + 2 H2O', 'H2 + O2 -> H2O2']))
    rs2 = ReactionSystem.from_string('\n'.join(['2 NH3 -> N2 + 3 H2']))
    rs3 = rs1 + rs2
    assert rs1 == rs1
    assert rs1 != rs2
    assert rs3 != rs1
    assert len(rs1.rxns) == 2 and len(rs2.rxns) == 1 and (len(rs3.rxns) == 3)
    for k in 'H2O2 O2 H2O H2 NH3 N2'.split():
        assert k in rs3.substances
    rs1 += rs2
    assert len(rs1.rxns) == 3 and len(rs2.rxns) == 1
    assert rs1 == rs3
    rs4 = ReactionSystem.from_string('H2O -> H+ + OH-; 1e-4')
    rs4 += [Reaction({'H+', 'OH-'}, {'H2O'}, 10000000000.0)]
    assert len(rs4.rxns) == 2
    assert rs4.rxns[0].reac == {'H2O': 1}
    assert rs4.rxns[1].reac == {'H+': 1, 'OH-': 1}
    res = rs4.rates({'H2O': 1, 'H+': 1e-07, 'OH-': 1e-07})
    for k in 'H2O H+ OH-'.split():
        assert abs(res[k]) < 1e-16
    rs5 = ReactionSystem.from_string('H3O+ -> H+ + H2O')
    rs6 = rs4 + rs5
    rs7 = rs6 + (Reaction.from_string('H+ + H2O -> H3O+'),)
    assert len(rs7.rxns) == 4
    with pytest.raises(ValueError):
        rs5 += (rs1, rs2)
    with pytest.raises(ValueError):
        rs5 + (rs1, rs2)
    rs1 = ReactionSystem.from_string('O2 + H2 -> H2O2')
    rs1.substances['H2O2'].data['D'] = 123
    rs2 = ReactionSystem.from_string('H2O2 -> 2 OH')
    rs2.substances['H2O2'].data['D'] = 456
    rs2.substances['OH'].data['D'] = 789
    rs3 = rs2 + rs1
    assert rs3.substances['H2O2'].data['D'] == 123 and rs3.substances['OH'].data['D'] == 789
    assert rs3.rxns[0].reac == {'H2O2': 1}
    assert rs3.rxns[1].reac == {'O2': 1, 'H2': 1}
    assert len(rs3.rxns) == 2
    rs2 += rs1
    assert rs2.substances['H2O2'].data['D'] == 123 and rs2.substances['OH'].data['D'] == 789
    assert rs2.rxns[0].reac == {'H2O2': 1}
    assert rs2.rxns[1].reac == {'O2': 1, 'H2': 1}
    assert len(rs2.rxns) == 2