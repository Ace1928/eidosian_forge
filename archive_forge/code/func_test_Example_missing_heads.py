import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_missing_heads():
    vocab = Vocab()
    words = ['I', 'like', 'London', 'and', 'Berlin', '.']
    deps = ['nsubj', 'ROOT', 'dobj', None, 'conj', 'punct']
    heads = [1, 1, 1, None, 2, 1]
    annots = {'words': words, 'heads': heads, 'deps': deps}
    predicted = Doc(vocab, words=words)
    example = Example.from_dict(predicted, annots)
    parsed_heads = [t.head.i for t in example.reference]
    assert parsed_heads[0] == heads[0]
    assert parsed_heads[1] == heads[1]
    assert parsed_heads[2] == heads[2]
    assert parsed_heads[4] == heads[4]
    assert parsed_heads[5] == heads[5]
    expected = [True, True, True, False, True, True]
    assert [t.has_head() for t in example.reference] == expected
    expected = [True, False, False, False, False, False]
    assert example.get_aligned_sent_starts() == expected