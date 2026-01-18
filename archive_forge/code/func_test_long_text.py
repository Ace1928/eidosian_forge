import pytest
from spacy.lang.ta import Tamil
@pytest.mark.parametrize('text, num_tokens', [(TAMIL_BASIC_TOKENIZER_SENTENCIZER_TEST_TEXT, 23 + 90)])
def test_long_text(ta_tokenizer, text, num_tokens):
    tokens = ta_tokenizer(text)
    assert len(tokens) == num_tokens