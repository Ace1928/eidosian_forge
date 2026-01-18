import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def ja_tokenizer():
    pytest.importorskip('sudachipy')
    return get_lang_class('ja')().tokenizer