import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct_open,punct_close', PUNCT_PAIRED)
@pytest.mark.parametrize('text', ['Hello'])
def test_en_tokenizer_splits_open_close_punct(en_tokenizer, punct_open, punct_close, text):
    tokens = en_tokenizer(punct_open + text + punct_close)
    assert len(tokens) == 3
    assert tokens[0].text == punct_open
    assert tokens[1].text == text
    assert tokens[2].text == punct_close