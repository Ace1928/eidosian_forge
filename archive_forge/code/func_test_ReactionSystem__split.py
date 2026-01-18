from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, 'numpy')
def test_ReactionSystem__split():
    a = '\n    2 H2 +  O2 -> 2 H2O     ; 1e-3\n           H2O -> H+ + OH-  ; 1e-4/55.35\n      H+ + OH- -> H2O       ; 1e10\n        2 H2O  -> 2 H2 + O2'
    b = '\n        2 N    -> N2'
    c = '\n        2 ClBr -> Cl2 + Br2\n    '
    rsys1 = ReactionSystem.from_string(a + b + c)
    res = rsys1.split()
    ref = list(map(ReactionSystem.from_string, [a, b, c]))
    for rs in chain(res, ref):
        rs.sort_substances_inplace()
    res1a, res1b, res1c = res
    ref1a, ref1b, ref1c = ref
    assert res1a == ref1a
    assert res1b == ref1b
    assert res1c == ref1c
    assert res1c != ref1a
    assert rsys1.categorize_substances() == dict(accumulated={'N2', 'Cl2', 'Br2'}, depleted={'N', 'ClBr'}, unaffected=set(), nonparticipating=set())