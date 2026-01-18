import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='module')
def la_tokenizer():
    return get_lang_class('la')().tokenizer