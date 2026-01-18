import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_array_idx(en_vocab):
    """Test that Doc.to_array can retrieve token start indices"""
    words = ['An', 'example', 'sentence']
    offsets = Doc(en_vocab, words=words).to_array('IDX')
    assert offsets[0] == 0
    assert offsets[1] == 3
    assert offsets[2] == 11