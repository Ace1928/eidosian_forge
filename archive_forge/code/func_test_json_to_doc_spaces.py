import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_spaces():
    """Test that Doc.from_json() preserves spaces correctly."""
    doc = spacy.blank('en')('This is just brilliant.')
    json_doc = doc.to_json()
    new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
    assert doc.text == new_doc.text