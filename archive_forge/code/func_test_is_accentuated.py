import logging
import pytest
from charset_normalizer.utils import is_accentuated, cp_similarity, set_logging_handler
@pytest.mark.parametrize('character, expected_is_accentuated', [('é', True), ('è', True), ('à', True), ('À', True), ('Ù', True), ('ç', True), ('a', False), ('€', False), ('&', False), ('Ö', True), ('ü', True), ('ê', True), ('Ñ', True), ('Ý', True), ('Ω', False), ('ø', False), ('Ё', False)])
def test_is_accentuated(character, expected_is_accentuated):
    assert is_accentuated(character) is expected_is_accentuated, 'is_accentuated behavior incomplete'