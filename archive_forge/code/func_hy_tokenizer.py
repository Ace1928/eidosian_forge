import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def hy_tokenizer():
    return get_lang_class('hy')().tokenizer