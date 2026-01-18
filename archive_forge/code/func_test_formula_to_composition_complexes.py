import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('Al2(SO4)3', {8: 12, 13: 2, 16: 3}), ('Al2(SO4)3(s)', {8: 12, 13: 2, 16: 3}), ('Al2(SO4)3(aq)', {8: 12, 13: 2, 16: 3}), ('K4[Fe(CN)6]', {6: 6, 7: 6, 19: 4, 26: 1}), ('K4[Fe(CN)6](s)', {6: 6, 7: 6, 19: 4, 26: 1}), ('K4[Fe(CN)6](aq)', {6: 6, 7: 6, 19: 4, 26: 1}), ('[Fe(H2O)6][Fe(CN)6]..19H2O', {1: 50, 6: 6, 7: 6, 8: 25, 26: 2}), ('[Fe(H2O)6][Fe(CN)6]..19H2O(s)', {1: 50, 6: 6, 7: 6, 8: 25, 26: 2}), ('[Fe(H2O)6][Fe(CN)6]..19H2O(aq)', {1: 50, 6: 6, 7: 6, 8: 25, 26: 2}), ('[Fe(CN)6]-3', {0: -3, 6: 6, 7: 6, 26: 1}), ('[Fe(CN)6]-3(aq)', {0: -3, 6: 6, 7: 6, 26: 1}), ('Ag[NH3]+', {0: 1, 1: 3, 7: 1, 47: 1}), ('[Ni(NH3)6]+2', {0: 2, 1: 18, 7: 6, 28: 1}), ('[PtCl6]-2', {0: -2, 17: 6, 78: 1})])
@requires(parsing_library)
def test_formula_to_composition_complexes(species, composition):
    assert formula_to_composition(species) == composition