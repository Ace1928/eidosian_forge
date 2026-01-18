import logging
import pytest
from charset_normalizer.utils import is_accentuated, cp_similarity, set_logging_handler
@pytest.mark.parametrize('cp_name_a, cp_name_b, expected_is_similar', [('cp1026', 'cp1140', True), ('cp1140', 'cp1026', True), ('latin_1', 'cp1252', True), ('latin_1', 'iso8859_4', True), ('latin_1', 'cp1251', False), ('cp1251', 'mac_turkish', False)])
def test_cp_similarity(cp_name_a, cp_name_b, expected_is_similar):
    is_similar = cp_similarity(cp_name_a, cp_name_b) >= 0.8
    assert is_similar is expected_is_similar, 'cp_similarity is broken'