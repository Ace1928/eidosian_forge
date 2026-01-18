import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json_underscore_error_attr(doc):
    """Test that Doc.to_json() raises an error if a custom attribute doesn't
    exist in the ._ space."""
    with pytest.raises(ValueError):
        doc.to_json(underscore=['json_test3'])