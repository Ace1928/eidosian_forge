from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library)
def test_ReactionSystem__missing_substances_from_keys():
    r1 = Reaction({'H2O'}, {'H+', 'OH-'})
    with pytest.raises(ValueError):
        ReactionSystem([r1], substances={'H2O': Substance.from_formula('H2O')})
    kw = dict(missing_substances_from_keys=True, substance_factory=Substance.from_formula)
    rs = ReactionSystem([r1], substances={'H2O': Substance.from_formula('H2O')}, **kw)
    assert rs.substances['OH-'].composition == {0: -1, 1: 1, 8: 1}