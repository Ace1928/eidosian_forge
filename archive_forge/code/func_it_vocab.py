import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def it_vocab():
    return get_lang_class('it')().vocab