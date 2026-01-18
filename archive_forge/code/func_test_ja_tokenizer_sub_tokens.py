import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,sub_tokens_list_b,sub_tokens_list_c', SUB_TOKEN_TESTS)
def test_ja_tokenizer_sub_tokens(ja_tokenizer, text, sub_tokens_list_b, sub_tokens_list_c):
    nlp_a = Japanese.from_config({'nlp': {'tokenizer': {'split_mode': 'A'}}})
    nlp_b = Japanese.from_config({'nlp': {'tokenizer': {'split_mode': 'B'}}})
    nlp_c = Japanese.from_config({'nlp': {'tokenizer': {'split_mode': 'C'}}})
    assert ja_tokenizer(text).user_data.get('sub_tokens') is None
    assert nlp_a(text).user_data.get('sub_tokens') is None
    assert nlp_b(text).user_data['sub_tokens'] == sub_tokens_list_b
    assert nlp_c(text).user_data['sub_tokens'] == sub_tokens_list_c