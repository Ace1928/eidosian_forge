import pytest
from ..periodic import (
from ..testing import requires
from ..parsing import formula_to_composition, parsing_library
@requires(parsing_library)
def test_mass_from_composition__formula():
    mass = mass_from_composition(formula_to_composition('NaF'))
    assert abs(41.988172443 - mass) < 1e-07
    Fminus = mass_from_composition(formula_to_composition('F-'))
    assert abs(Fminus - 18.998403163 - 0.0005489) < 1e-07