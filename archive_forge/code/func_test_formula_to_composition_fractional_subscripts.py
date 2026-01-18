import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('Ca2.832Fe0.6285Mg5.395(CO3)6', {6: 6, 8: 18, 12: 5.395, 20: 2.832, 26: 0.6285}), ('Ca2.832Fe0.6285Mg5.395(CO3)6(s)', {6: 6, 8: 18, 12: 5.395, 20: 2.832, 26: 0.6285}), ('Ca2.832Fe0.6285Mg5.395(CO3)6..8H2O(s)', {1: 16, 6: 6, 8: 26, 12: 5.395, 20: 2.832, 26: 0.6285})])
@requires(parsing_library)
def test_formula_to_composition_fractional_subscripts(species, composition):
    assert formula_to_composition(species) == composition