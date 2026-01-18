import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_validation_error(doc):
    """Test that Doc.from_json() raises an exception when validating invalid input."""
    doc_json = doc.to_json()
    doc_json.pop('tokens')
    with pytest.raises(ValueError):
        Doc(doc.vocab).from_json(doc_json, validate=True)