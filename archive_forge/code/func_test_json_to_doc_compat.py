import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
def test_json_to_doc_compat(doc, doc_json):
    new_doc = Doc(doc.vocab).from_json(doc_json, validate=True)
    new_tokens = [token for token in new_doc]
    assert new_doc.text == doc.text == 'c d e '
    assert len(new_tokens) == len([token for token in doc]) == 3
    assert new_tokens[0].pos == doc[0].pos
    assert new_tokens[0].tag == doc[0].tag
    assert new_tokens[0].dep == doc[0].dep
    assert new_tokens[0].head.idx == doc[0].head.idx
    assert new_tokens[0].lemma == doc[0].lemma
    assert len(new_doc.ents) == 1
    assert new_doc.ents[0].start == 1
    assert new_doc.ents[0].end == 2
    assert new_doc.ents[0].label_ == 'ORG'