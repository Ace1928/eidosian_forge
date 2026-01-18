import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('CO2(aq)', {6: 1, 8: 2}), ('H2O(aq)', {1: 2, 8: 1})])
@requires(parsing_library)
def test_formula_to_composition_state_not_in_suffixes(species, composition):
    """Should parse species without state in suffixes."""
    assert formula_to_composition(species, suffixes=('(g)', '(l)', '(s)')) == composition