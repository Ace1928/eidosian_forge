import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def ko_tokenizer_tokenizer():
    config = {'nlp': {'tokenizer': {'@tokenizers': 'spacy.Tokenizer.v1'}}}
    nlp = get_lang_class('ko').from_config(config)
    return nlp.tokenizer