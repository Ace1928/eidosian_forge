import pytest
from spacy.util import get_lang_class
@pytest.mark.parametrize('lang', LANGUAGES)
def test_lang_initialize(lang, capfd):
    """Test that languages can be initialized."""
    nlp = get_lang_class(lang)()
    doc = nlp('test')
    captured = capfd.readouterr()
    assert not captured.out