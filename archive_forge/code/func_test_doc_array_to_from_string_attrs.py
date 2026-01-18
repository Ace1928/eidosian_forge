import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
@pytest.mark.parametrize('attrs', [['ORTH', 'SHAPE'], 'IS_ALPHA'])
def test_doc_array_to_from_string_attrs(en_vocab, attrs):
    """Test that both Doc.to_array and Doc.from_array accept string attrs,
    as well as single attrs and sequences of attrs.
    """
    words = ['An', 'example', 'sentence']
    doc = Doc(en_vocab, words=words)
    Doc(en_vocab, words=words).from_array(attrs, doc.to_array(attrs))