import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_from_dict_sentences():
    vocab = Vocab()
    predicted = Doc(vocab, words=['One', 'sentence', '.', 'one', 'more'])
    annots = {'sent_starts': [1, 0, 0, 1, 0]}
    ex = Example.from_dict(predicted, annots)
    assert len(list(ex.reference.sents)) == 2
    predicted = Doc(vocab, words=['One', 'sentence', 'not', 'one', 'more'])
    annots = {'sent_starts': [1, -1, 0, 0, 0]}
    ex = Example.from_dict(predicted, annots)
    assert len(list(ex.reference.sents)) == 1