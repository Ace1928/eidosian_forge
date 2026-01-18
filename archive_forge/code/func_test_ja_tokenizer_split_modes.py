import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,len_a,len_b,len_c', [('選挙管理委員会', 4, 3, 1), ('客室乗務員', 3, 2, 1), ('労働者協同組合', 4, 3, 1), ('機能性食品', 3, 2, 1)])
def test_ja_tokenizer_split_modes(ja_tokenizer, text, len_a, len_b, len_c):
    nlp_a = Japanese.from_config({'nlp': {'tokenizer': {'split_mode': 'A'}}})
    nlp_b = Japanese.from_config({'nlp': {'tokenizer': {'split_mode': 'B'}}})
    nlp_c = Japanese.from_config({'nlp': {'tokenizer': {'split_mode': 'C'}}})
    assert len(ja_tokenizer(text)) == len_a
    assert len(nlp_a(text)) == len_a
    assert len(nlp_b(text)) == len_b
    assert len(nlp_c(text)) == len_c