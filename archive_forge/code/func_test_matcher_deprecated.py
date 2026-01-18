import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_deprecated(matcher):
    doc = Doc(matcher.vocab, words=['hello', 'world'])
    with pytest.warns(DeprecationWarning) as record:
        for _ in matcher.pipe([doc]):
            pass
        assert record.list
        assert 'spaCy v3.0' in str(record.list[0].message)