import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text', ["Hello''"])
def test_en_tokenizer_splits_double_end_quote(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 2
    tokens_punct = en_tokenizer("''")
    assert len(tokens_punct) == 1