import pytest
from spacy.tokens import Doc
from ...util import apply_transition_sequence
@pytest.mark.parametrize('words', [['A', 'test', 'sentence']])
@pytest.mark.parametrize('punct', ['.', '!', '?', ''])
def test_en_sbd_single_punct(en_vocab, words, punct):
    heads = [2, 2, 2, 2] if punct else [2, 2, 2]
    deps = ['dep'] * len(heads)
    words = [*words, punct] if punct else words
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert len(doc) == 4 if punct else 3
    assert len(list(doc.sents)) == 1
    assert sum((len(sent) for sent in doc.sents)) == len(doc)