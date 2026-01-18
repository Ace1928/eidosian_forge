import pytest
from spacy.language import Language
from spacy.pipeline.functions import merge_subtokens
from spacy.tokens import Doc, Span
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.fixture
def doc2(en_vocab):
    words = ['I', 'like', 'New', 'York', 'in', 'Autumn', '.']
    heads = [1, 1, 3, 1, 1, 4, 1]
    tags = ['PRP', 'IN', 'NNP', 'NNP', 'IN', 'NNP', '.']
    pos = ['PRON', 'VERB', 'PROPN', 'PROPN', 'ADP', 'PROPN', 'PUNCT']
    deps = ['ROOT', 'prep', 'compound', 'pobj', 'prep', 'pobj', 'punct']
    doc = Doc(en_vocab, words=words, heads=heads, tags=tags, pos=pos, deps=deps)
    doc.ents = [Span(doc, 2, 4, label='GPE')]
    return doc