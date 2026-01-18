import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_doc_to_json(doc):
    json_doc = doc.to_json()
    assert json_doc['text'] == 'c d e '
    assert len(json_doc['tokens']) == 3
    assert json_doc['tokens'][0]['pos'] == 'VERB'
    assert json_doc['tokens'][0]['tag'] == 'VBP'
    assert json_doc['tokens'][0]['dep'] == 'ROOT'
    assert len(json_doc['ents']) == 1
    assert json_doc['ents'][0]['start'] == 2
    assert json_doc['ents'][0]['end'] == 3
    assert json_doc['ents'][0]['label'] == 'ORG'
    assert len(schemas.validate(schemas.DocJSONSchema, json_doc)) == 0
    assert srsly.json_loads(srsly.json_dumps(json_doc)) == json_doc