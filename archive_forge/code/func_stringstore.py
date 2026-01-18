import pytest
from spacy.strings import StringStore
@pytest.fixture
def stringstore():
    return StringStore()