import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
@pytest.fixture()
def doc_json():
    return {'text': 'c d e ', 'ents': [{'start': 2, 'end': 3, 'label': 'ORG'}], 'sents': [{'start': 0, 'end': 5}], 'tokens': [{'id': 0, 'start': 0, 'end': 1, 'tag': 'VBP', 'pos': 'VERB', 'morph': 'Feat1=A', 'dep': 'ROOT', 'head': 0}, {'id': 1, 'start': 2, 'end': 3, 'tag': 'NN', 'pos': 'NOUN', 'morph': 'Feat1=B', 'dep': 'dobj', 'head': 0}, {'id': 2, 'start': 4, 'end': 5, 'tag': 'NN', 'pos': 'NOUN', 'morph': 'Feat1=A|Feat2=D', 'dep': 'dobj', 'head': 1}]}