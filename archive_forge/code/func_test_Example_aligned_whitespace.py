import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_aligned_whitespace(en_vocab):
    words = ['a', ' ', 'b']
    tags = ['A', 'SPACE', 'B']
    predicted = Doc(en_vocab, words=words)
    reference = Doc(en_vocab, words=words, tags=tags)
    example = Example(predicted, reference)
    assert example.get_aligned('TAG', as_string=True) == tags