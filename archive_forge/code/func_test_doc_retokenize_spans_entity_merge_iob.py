import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_entity_merge_iob(en_vocab):
    words = ['a', 'b', 'c', 'd', 'e']
    doc = Doc(Vocab(), words=words)
    doc.ents = [(doc.vocab.strings.add('ent-abc'), 0, 3), (doc.vocab.strings.add('ent-d'), 3, 4)]
    assert doc[0].ent_iob_ == 'B'
    assert doc[1].ent_iob_ == 'I'
    assert doc[2].ent_iob_ == 'I'
    assert doc[3].ent_iob_ == 'B'
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2])
    assert len(doc) == len(words) - 1
    assert doc[0].ent_iob_ == 'B'
    assert doc[1].ent_iob_ == 'I'
    words = ['a', 'b', 'c', 'd', 'e']
    doc = Doc(Vocab(), words=words)
    with doc.retokenize() as retokenizer:
        attrs = {'ent_type': 'ent-abc', 'ent_iob': 1}
        retokenizer.merge(doc[0:3], attrs=attrs)
        retokenizer.merge(doc[3:5], attrs=attrs)
    assert doc[0].ent_iob_ == 'B'
    assert doc[1].ent_iob_ == 'I'
    words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    doc = Doc(Vocab(), words=words)
    doc.ents = [(doc.vocab.strings.add('ent-de'), 3, 5), (doc.vocab.strings.add('ent-fg'), 5, 7)]
    assert doc[3].ent_iob_ == 'B'
    assert doc[4].ent_iob_ == 'I'
    assert doc[5].ent_iob_ == 'B'
    assert doc[6].ent_iob_ == 'I'
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[2:4])
        retokenizer.merge(doc[4:6])
        retokenizer.merge(doc[7:9])
    assert len(doc) == 6
    assert doc[3].ent_iob_ == 'B'
    assert doc[3].ent_type_ == 'ent-de'
    assert doc[4].ent_iob_ == 'B'
    assert doc[4].ent_type_ == 'ent-fg'
    words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    heads = [0, 0, 3, 0, 0, 0, 5, 0, 0]
    ents = ['O'] * len(words)
    ents[3] = 'B-ent-de'
    ents[4] = 'I-ent-de'
    ents[5] = 'B-ent-fg'
    ents[6] = 'I-ent-fg'
    deps = ['dep'] * len(words)
    en_vocab.strings.add('ent-de')
    en_vocab.strings.add('ent-fg')
    en_vocab.strings.add('dep')
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps, ents=ents)
    assert doc[2:4].root == doc[3]
    assert doc[4:6].root == doc[4]
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[2:4])
        retokenizer.merge(doc[4:6])
        retokenizer.merge(doc[7:9])
    assert len(doc) == 6
    assert doc[2].ent_iob_ == 'B'
    assert doc[2].ent_type_ == 'ent-de'
    assert doc[3].ent_iob_ == 'I'
    assert doc[3].ent_type_ == 'ent-de'
    assert doc[4].ent_iob_ == 'B'
    assert doc[4].ent_type_ == 'ent-fg'
    words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    heads = [0, 0, 3, 4, 0, 0, 5, 0, 0]
    ents = ['O'] * len(words)
    ents[3] = 'B-ent-de'
    ents[4] = 'I-ent-de'
    ents[5] = 'B-ent-de'
    ents[6] = 'I-ent-de'
    deps = ['dep'] * len(words)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps, ents=ents)
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[3:5])
        retokenizer.merge(doc[5:7])
    assert len(doc) == 7
    assert doc[3].ent_iob_ == 'B'
    assert doc[3].ent_type_ == 'ent-de'
    assert doc[4].ent_iob_ == 'B'
    assert doc[4].ent_type_ == 'ent-de'