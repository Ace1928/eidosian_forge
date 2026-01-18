import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, composition', [('BaCl2', {17: 2, 56: 1}), ('BaCl2(s)', {17: 2, 56: 1}), ('BaCl2..2H2O(s)', {1: 4, 8: 2, 17: 2, 56: 1}), ('Na2CO3..7H2O(s)', {11: 2, 6: 1, 8: 10, 1: 14}), ('NaCl', {11: 1, 17: 1}), ('NaCl(s)', {11: 1, 17: 1}), ('Ni', {28: 1}), ('NI', {7: 1, 53: 1}), ('KF', {9: 1, 19: 1})])
@requires(parsing_library)
def test_formula_to_composition_ionic_compounds(species, composition):
    assert formula_to_composition(species) == composition