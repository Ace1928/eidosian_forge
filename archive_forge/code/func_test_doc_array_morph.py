import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_array_morph(en_vocab):
    words = ['Eat', 'blue', 'ham']
    morph = ['Feat=V', 'Feat=J', 'Feat=N']
    doc = Doc(en_vocab, words=words, morphs=morph)
    assert morph[0] == str(doc[0].morph)
    assert morph[1] == str(doc[1].morph)
    assert morph[2] == str(doc[2].morph)
    feats_array = doc.to_array((ORTH, MORPH))
    assert feats_array[0][1] == doc[0].morph.key
    assert feats_array[1][1] == doc[1].morph.key
    assert feats_array[2][1] == doc[2].morph.key