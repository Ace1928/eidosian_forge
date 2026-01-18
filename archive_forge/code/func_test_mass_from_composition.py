import pytest
from ..periodic import (
from ..testing import requires
from ..parsing import formula_to_composition, parsing_library
def test_mass_from_composition():
    mass = mass_from_composition({11: 1, 9: 1})
    assert abs(41.988172443 - mass) < 1e-07