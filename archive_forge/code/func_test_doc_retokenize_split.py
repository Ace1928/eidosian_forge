import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_split(en_vocab):
    words = ['LosAngeles', 'start', '.']
    heads = [1, 2, 2]
    deps = ['dep'] * len(heads)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    assert len(doc) == 3
    assert len(str(doc)) == 19
    assert doc[0].head.text == 'start'
    assert doc[1].head.text == '.'
    with doc.retokenize() as retokenizer:
        retokenizer.split(doc[0], ['Los', 'Angeles'], [(doc[0], 1), doc[1]], attrs={'tag': ['NNP'] * 2, 'lemma': ['Los', 'Angeles'], 'ent_type': ['GPE'] * 2, 'morph': ['Number=Sing'] * 2})
    assert len(doc) == 4
    assert doc[0].text == 'Los'
    assert doc[0].head.text == 'Angeles'
    assert doc[0].idx == 0
    assert str(doc[0].morph) == 'Number=Sing'
    assert doc[1].idx == 3
    assert doc[1].text == 'Angeles'
    assert doc[1].head.text == 'start'
    assert str(doc[1].morph) == 'Number=Sing'
    assert doc[2].text == 'start'
    assert doc[2].head.text == '.'
    assert doc[3].text == '.'
    assert doc[3].head.text == '.'
    assert len(str(doc)) == 19