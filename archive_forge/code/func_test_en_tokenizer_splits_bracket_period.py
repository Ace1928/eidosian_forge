import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
def test_en_tokenizer_splits_bracket_period(en_tokenizer):
    text = '(And a 6a.m. run through Washington Park).'
    tokens = en_tokenizer(text)
    assert tokens[len(tokens) - 1].text == '.'