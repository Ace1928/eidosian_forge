import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_entity_split_iob():
    words = ['abc', 'd', 'e']
    doc = Doc(Vocab(), words=words)
    doc.ents = [(doc.vocab.strings.add('ent-abcd'), 0, 2)]
    assert doc[0].ent_iob_ == 'B'
    assert doc[1].ent_iob_ == 'I'
    with doc.retokenize() as retokenizer:
        retokenizer.split(doc[0], ['a', 'b', 'c'], [(doc[0], 1), (doc[0], 2), doc[1]])
    assert doc[0].ent_iob_ == 'B'
    assert doc[1].ent_iob_ == 'I'
    assert doc[2].ent_iob_ == 'I'
    assert doc[3].ent_iob_ == 'I'