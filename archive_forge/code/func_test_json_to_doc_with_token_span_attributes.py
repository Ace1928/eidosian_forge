import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_with_token_span_attributes(doc):
    Doc.set_extension('json_test1', default=False)
    Doc.set_extension('json_test2', default=False)
    Token.set_extension('token_test', default=False)
    Span.set_extension('span_test', default=False)
    doc._.json_test1 = 'hello world'
    doc._.json_test2 = [1, 2, 3]
    doc[0:1]._.span_test = 'span_attribute'
    doc[0:2]._.span_test = 'span_attribute_2'
    doc[0]._.token_test = 117
    doc[1]._.token_test = 118
    json_doc = doc.to_json(underscore=['json_test1', 'json_test2', 'token_test', 'span_test'])
    json_doc = srsly.json_loads(srsly.json_dumps(json_doc))
    new_doc = Doc(doc.vocab).from_json(json_doc, validate=True)
    assert all([new_doc.has_extension(f'json_test{i}') for i in range(1, 3)])
    assert new_doc._.json_test1 == 'hello world'
    assert new_doc._.json_test2 == [1, 2, 3]
    assert new_doc[0]._.token_test == 117
    assert new_doc[1]._.token_test == 118
    assert new_doc[0:1]._.span_test == 'span_attribute'
    assert new_doc[0:2]._.span_test == 'span_attribute_2'
    assert new_doc.user_data == doc.user_data
    assert new_doc.to_bytes(exclude=['user_data']) == doc.to_bytes(exclude=['user_data'])