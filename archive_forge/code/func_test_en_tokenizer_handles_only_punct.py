import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text', ['(', '((', '<'])
def test_en_tokenizer_handles_only_punct(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == len(text)