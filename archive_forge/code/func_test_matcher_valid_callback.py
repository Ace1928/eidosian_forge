import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.filterwarnings('ignore:\\[W036')
def test_matcher_valid_callback(en_vocab):
    """Test that on_match can only be None or callable."""
    matcher = Matcher(en_vocab)
    with pytest.raises(ValueError):
        matcher.add('TEST', [[{'TEXT': 'test'}]], on_match=[])
    matcher(Doc(en_vocab, words=['test']))