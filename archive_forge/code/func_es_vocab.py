import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def es_vocab():
    return get_lang_class('es')().vocab