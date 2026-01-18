import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json_with_custom_user_data(doc):
    Doc.set_extension('json_test', default=False)
    Token.set_extension('token_test', default=False)
    Span.set_extension('span_test', default=False)
    doc._.json_test = 'hello world'
    doc[0:1]._.span_test = 'span_attribute'
    doc[0]._.token_test = 117
    json_doc = doc.to_json(underscore=['json_test', 'token_test', 'span_test'])
    doc.user_data['user_data_test'] = 10
    doc.user_data['user_data_test2', True] = 10
    assert '_' in json_doc
    assert json_doc['_']['json_test'] == 'hello world'
    assert 'underscore_token' in json_doc
    assert 'underscore_span' in json_doc
    assert json_doc['underscore_token']['token_test'][0]['value'] == 117
    assert json_doc['underscore_span']['span_test'][0]['value'] == 'span_attribute'
    assert len(schemas.validate(schemas.DocJSONSchema, json_doc)) == 0
    assert srsly.json_loads(srsly.json_dumps(json_doc)) == json_doc