import pytest
from spacy.language import Language
from spacy.pipeline.functions import merge_subtokens
from spacy.tokens import Doc, Span
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_factories_merge_ents(doc2):
    assert len(doc2) == 7
    assert len(list(doc2.ents)) == 1
    nlp = Language()
    merge_entities = nlp.create_pipe('merge_entities')
    merge_entities(doc2)
    assert len(doc2) == 6
    assert len(list(doc2.ents)) == 1
    assert doc2[2].text == 'New York'