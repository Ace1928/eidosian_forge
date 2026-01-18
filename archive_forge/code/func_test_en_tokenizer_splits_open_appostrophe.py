import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text', ["'The"])
def test_en_tokenizer_splits_open_appostrophe(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == "'"