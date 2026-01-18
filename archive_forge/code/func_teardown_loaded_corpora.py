import pytest
from nltk.corpus.reader import CorpusReader
@pytest.fixture(scope='module', autouse=True)
def teardown_loaded_corpora():
    """
    After each test session ends (either doctest or unit test),
    unload any loaded corpora
    """
    yield
    import nltk.corpus
    for name in dir(nltk.corpus):
        obj = getattr(nltk.corpus, name, None)
        if isinstance(obj, CorpusReader) and hasattr(obj, '_unload'):
            obj._unload()