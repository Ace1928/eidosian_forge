import pytest
import nltk.data
def test_find_raises_exception():
    with pytest.raises(LookupError):
        nltk.data.find('no_such_resource/foo')