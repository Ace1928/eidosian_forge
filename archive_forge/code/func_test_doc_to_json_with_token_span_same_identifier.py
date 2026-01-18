import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json_with_token_span_same_identifier(doc):
    Doc.set_extension('my_ext', default=False)
    Token.set_extension('my_ext', default=False)
    Span.set_extension('my_ext', default=False)
    doc._.my_ext = 'hello world'
    doc[0:1]._.my_ext = 'span_attribute'
    doc[0]._.my_ext = 117
    json_doc = doc.to_json(underscore=['my_ext'])
    assert '_' in json_doc
    assert json_doc['_']['my_ext'] == 'hello world'
    assert 'underscore_token' in json_doc
    assert 'underscore_span' in json_doc
    assert json_doc['underscore_token']['my_ext'][0]['value'] == 117
    assert json_doc['underscore_span']['my_ext'][0]['value'] == 'span_attribute'
    assert len(schemas.validate(schemas.DocJSONSchema, json_doc)) == 0
    assert srsly.json_loads(srsly.json_dumps(json_doc)) == json_doc