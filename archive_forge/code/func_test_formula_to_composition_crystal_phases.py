import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('alpha-FeOOH(s)', {1: 1, 8: 2, 26: 1}), ('epsilon-Zn(OH)2(s)', {1: 2, 8: 2, 30: 1})])
@requires(parsing_library)
def test_formula_to_composition_crystal_phases(species, composition):
    assert formula_to_composition(species) == composition