import pytest
from mock import Mock
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
def test_create_doc_underscore():
    doc = Mock()
    doc.doc = doc
    uscore = Underscore(Underscore.doc_extensions, doc)
    assert uscore._doc is doc
    assert uscore._start is None
    assert uscore._end is None