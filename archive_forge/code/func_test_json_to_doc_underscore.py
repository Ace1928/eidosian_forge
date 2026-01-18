import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_underscore(doc):
    Doc.set_extension('json_test1', default=False)
    Doc.set_extension('json_test2', default=False)
    doc._.json_test1 = 'hello world'
    doc._.json_test2 = [1, 2, 3]
    json_doc = doc.to_json(underscore=['json_test1', 'json_test2'])
    new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
    assert all([new_doc.has_extension(f'json_test{i}') for i in range(1, 3)])
    assert new_doc._.json_test1 == 'hello world'
    assert new_doc._.json_test2 == [1, 2, 3]
    assert doc.to_bytes() == new_doc.to_bytes()