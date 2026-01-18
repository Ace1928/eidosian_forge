from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library)
def test_ReactionSystem__concatenate():
    rs1 = ReactionSystem.from_string("\n    H + OH -> H2O; 1e10; name='rs1a'\n    2 H2O2 -> 2 H2O + O2; 1e-7; name='rs1b'\n")
    rs2 = ReactionSystem.from_string("\n    H + OH -> H2O; 1e11; name='rs2a'\n    H2O2 -> H2 + O2; 1e-9; name='rs2b'\n")
    rs, skipped = ReactionSystem.concatenate([rs1, rs2])
    sr, = skipped.rxns
    assert sr.name == 'rs2a'
    assert rs.rxns[-1].name == 'rs2b'