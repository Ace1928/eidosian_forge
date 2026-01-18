import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text,punct', [("(can't", '(')])
def test_en_tokenizer_splits_pre_punct_regex(text, punct):
    en_search_prefixes = compile_prefix_regex(TOKENIZER_PREFIXES).search
    match = en_search_prefixes(text)
    assert match.group() == punct