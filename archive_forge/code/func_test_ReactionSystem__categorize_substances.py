from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, 'numpy')
def test_ReactionSystem__categorize_substances():
    rsys1 = ReactionSystem.from_string('\n    2 H2 +  O2 -> 2 H2O     ; 1e-3\n           H2O -> H+ + OH-  ; 1e-4/55.35\n      H+ + OH- -> H2O       ; 1e10\n         2 H2O -> 2 H2 + O2\n    ')
    assert all((not s for s in rsys1.categorize_substances().values()))
    rsys2 = ReactionSystem.from_string('\n'.join(['2 NH3 -> N2 + 3 H2', 'N2H4 -> N2 +   2  H2']))
    assert rsys2.categorize_substances() == dict(accumulated={'N2', 'H2'}, depleted={'NH3', 'N2H4'}, unaffected=set(), nonparticipating=set())
    rsys3 = ReactionSystem.from_string("H+ + OH- -> H2O; 'kf'")
    assert rsys3.categorize_substances() == dict(accumulated={'H2O'}, depleted={'H+', 'OH-'}, unaffected=set(), nonparticipating=set())
    rsys4 = ReactionSystem([Reaction({'H2': 2, 'O2': 1}, {'H2O': 2})], 'H2 O2 H2O N2 Ar')
    assert rsys4.categorize_substances() == dict(accumulated={'H2O'}, depleted={'H2', 'O2'}, unaffected=set(), nonparticipating={'N2', 'Ar'})
    rsys5 = ReactionSystem.from_string("\n    A -> B; MassAction(unique_keys=('k1',))\n    B + C -> A + C; MassAction(unique_keys=('k2',))\n    2 B -> B + C; MassAction(unique_keys=('k3',))\n    ", substance_factory=lambda formula: Substance(formula))
    assert rsys5.categorize_substances() == dict(accumulated={'C'}, depleted=set(), unaffected=set(), nonparticipating=set())
    rsys6 = ReactionSystem.from_string('H2O2 + Fe+3 + (H2O2) -> 2 H2O + O2 + Fe+3')
    assert rsys6.rxns[0].order() == 2
    assert rsys6.categorize_substances() == dict(accumulated={'H2O', 'O2'}, depleted={'H2O2'}, unaffected={'Fe+3'}, nonparticipating=set())