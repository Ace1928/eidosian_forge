import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_sents(doc, doc_without_deps):
    """Test that Doc.from_json() includes correct.sents."""
    for test_doc in (doc, doc_without_deps):
        json_doc = test_doc.to_json()
        new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
        assert [sent.text for sent in test_doc.sents] == [sent.text for sent in new_doc.sents]
        assert [token.is_sent_start for token in test_doc] == [token.is_sent_start for token in new_doc]