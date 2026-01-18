import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('text', ['Hello'])
def test_en_tokenizer_splits_close_punct(en_tokenizer, punct, text):
    tokens = en_tokenizer(text + punct)
    assert len(tokens) == 2
    assert tokens[0].text == text
    assert tokens[1].text == punct