import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('H2O', {1: 2, 8: 1}), ('((H2O)2OH)12', {1: 60, 8: 36}), ('PCl5', {15: 1, 17: 5})])
@requires(parsing_library)
def test_formula_to_composition_covalent_compounds(species, composition):
    assert formula_to_composition(species) == composition