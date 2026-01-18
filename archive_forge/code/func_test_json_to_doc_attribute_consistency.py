import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_attribute_consistency(doc):
    """Test that Doc.from_json() raises an exception if tokens don't all have the same set of properties."""
    doc_json = doc.to_json()
    doc_json['tokens'][1].pop('morph')
    with pytest.raises(ValueError):
        Doc(doc.vocab).from_json(doc_json)