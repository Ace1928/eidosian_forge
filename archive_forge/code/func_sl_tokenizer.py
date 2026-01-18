import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def sl_tokenizer():
    return get_lang_class('sl')().tokenizer