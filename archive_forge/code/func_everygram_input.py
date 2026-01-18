import pytest
from nltk.util import everygrams
@pytest.fixture
def everygram_input():
    """Form test data for tests."""
    return iter(['a', 'b', 'c'])