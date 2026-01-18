import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
def test_doc_from_array_heads_in_bounds(en_vocab):
    """Test that Doc.from_array doesn't set heads that are out of bounds."""
    words = ['This', 'is', 'a', 'sentence', '.']
    doc = Doc(en_vocab, words=words)
    for token in doc:
        token.head = doc[0]
    arr = doc.to_array(['HEAD'])
    doc_from_array = Doc(en_vocab, words=words)
    doc_from_array.from_array(['HEAD'], arr)
    arr = doc.to_array(['HEAD'])
    arr[0] = numpy.int32(-1).astype(numpy.uint64)
    doc_from_array = Doc(en_vocab, words=words)
    with pytest.raises(ValueError):
        doc_from_array.from_array(['HEAD'], arr)
    arr = doc.to_array(['HEAD'])
    arr[0] = numpy.int32(5).astype(numpy.uint64)
    doc_from_array = Doc(en_vocab, words=words)
    with pytest.raises(ValueError):
        doc_from_array.from_array(['HEAD'], arr)