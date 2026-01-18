import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_stringy_array_attr_of_token(en_vocab):
    doc = Doc(en_vocab, words=['An', 'example', 'sentence'])
    example = doc.vocab['example']
    assert example.orth != example.shape
    feats_array = doc.to_array((ORTH, SHAPE))
    feats_array_stringy = doc.to_array(('ORTH', 'SHAPE'))
    assert feats_array_stringy[0][0] == feats_array[0][0]
    assert feats_array_stringy[0][1] == feats_array[0][1]