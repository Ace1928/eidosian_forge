import pytest
import srsly
import spacy
from spacy import schemas
from spacy.tokens import Doc, Span, Token
from .test_underscore import clean_underscore  # noqa: F401
@pytest.fixture()
def doc_without_deps(en_vocab):
    words = ['c', 'd', 'e']
    pos = ['VERB', 'NOUN', 'NOUN']
    tags = ['VBP', 'NN', 'NN']
    ents = ['O', 'B-ORG', 'O']
    morphs = ['Feat1=A', 'Feat1=B', 'Feat1=A|Feat2=D']
    return Doc(en_vocab, words=words, pos=pos, tags=tags, ents=ents, morphs=morphs, sent_starts=[True, False, True])