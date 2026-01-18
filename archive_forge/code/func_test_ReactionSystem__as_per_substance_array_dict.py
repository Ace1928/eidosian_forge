from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(units_library)
def test_ReactionSystem__as_per_substance_array_dict():
    mol = default_units.mol
    m = default_units.metre
    M = default_units.molar
    rs = ReactionSystem([], [Substance('H2O')])
    c = rs.as_per_substance_array({'H2O': 1 * M}, unit=M)
    assert c.dimensionality == M.dimensionality
    assert abs(c[0] / (1000 * mol / m ** 3) - 1) < 1e-16
    c = rs.as_per_substance_array({'H2O': 1})
    with pytest.raises(KeyError):
        c = rs.as_per_substance_array({'H': 1})
    assert rs.as_per_substance_dict([42]) == {'H2O': 42}