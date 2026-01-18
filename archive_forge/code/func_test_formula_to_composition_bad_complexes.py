import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species', ['[Fe(CN)6)-3', '(Fe(CN)6]-3', '[Fe(CN]6]-3'])
@requires(parsing_library)
def test_formula_to_composition_bad_complexes(species):
    with pytest.raises(ParseException):
        formula_to_composition(species)