import pytest
from nltk.corpus.reader import CorpusReader
@pytest.fixture(autouse=True)
def mock_plot(mocker):
    """Disable matplotlib plotting in test code"""
    try:
        import matplotlib.pyplot as plt
        mocker.patch.object(plt, 'gca')
        mocker.patch.object(plt, 'show')
    except ImportError:
        pass