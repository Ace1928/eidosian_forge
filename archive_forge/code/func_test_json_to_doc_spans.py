import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_spans(doc):
    """Test that Doc.from_json() includes correct.spans."""
    doc.spans['test'] = [Span(doc, 0, 2, label='test'), Span(doc, 0, 1, label='test', kb_id=7)]
    json_doc = doc.to_json()
    new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
    assert len(new_doc.spans) == 1
    assert len(new_doc.spans['test']) == 2
    for i in range(2):
        assert new_doc.spans['test'][i].start == doc.spans['test'][i].start
        assert new_doc.spans['test'][i].end == doc.spans['test'][i].end
        assert new_doc.spans['test'][i].label == doc.spans['test'][i].label
        assert new_doc.spans['test'][i].kb_id == doc.spans['test'][i].kb_id