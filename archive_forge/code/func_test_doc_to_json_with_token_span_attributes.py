import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json_with_token_span_attributes(doc):
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
    doc.spans['span_group'] = [doc[0:1]]
    json_doc = doc.to_json(underscore=['json_test1', 'json_test2', 'token_test', 'span_test'])
    assert '_' in json_doc
    assert json_doc['_']['json_test1'] == 'hello world'
    assert json_doc['_']['json_test2'] == [1, 2, 3]
    assert 'underscore_token' in json_doc
    assert 'underscore_span' in json_doc
    assert json_doc['underscore_token']['token_test'][0]['value'] == 117
    assert json_doc['underscore_token']['token_test'][1]['value'] == 118
    assert json_doc['underscore_span']['span_test'][0]['value'] == 'span_attribute'
    assert json_doc['underscore_span']['span_test'][1]['value'] == 'span_attribute_2'
    assert len(schemas.validate(schemas.DocJSONSchema, json_doc)) == 0
    assert srsly.json_loads(srsly.json_dumps(json_doc)) == json_doc