import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('Cl-', {0: -1, 17: 1}), ('Fe(SCN)2+', {0: 1, 6: 2, 7: 2, 16: 2, 26: 1}), ('Fe(SCN)2+1', {0: 1, 6: 2, 7: 2, 16: 2, 26: 1}), ('Fe+3', {0: 3, 26: 1}), ('NH4+', {0: 1, 1: 4, 7: 1}), ('Na+', {0: 1, 11: 1}), ('Na+1', {0: 1, 11: 1}), ('OH-', {0: -1, 1: 1, 8: 1}), ('SO4-2(aq)', {0: -2, 8: 4, 16: 1})])
@requires(parsing_library)
def test_formula_to_composition_ions(species, composition):
    assert formula_to_composition(species) == composition