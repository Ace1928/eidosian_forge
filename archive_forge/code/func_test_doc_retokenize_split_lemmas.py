import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_split_lemmas(en_vocab):
    words = ['LosAngeles', 'start', '.']
    heads = [1, 2, 2]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    with doc.retokenize() as retokenizer:
        retokenizer.split(doc[0], ['Los', 'Angeles'], [(doc[0], 1), doc[1]])
    assert doc[0].lemma_ == ''
    assert doc[1].lemma_ == ''
    words = ['LosAngeles', 'start', '.']
    heads = [1, 2, 2]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    for t in doc:
        t.lemma_ = 'a'
    with doc.retokenize() as retokenizer:
        retokenizer.split(doc[0], ['Los', 'Angeles'], [(doc[0], 1), doc[1]])
    assert doc[0].lemma_ == 'Los'
    assert doc[1].lemma_ == 'Angeles'