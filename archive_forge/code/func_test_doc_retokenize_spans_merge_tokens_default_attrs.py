import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_merge_tokens_default_attrs(en_vocab):
    words = ['The', 'players', 'start', '.']
    lemmas = [t.lower() for t in words]
    heads = [1, 2, 2, 2]
    deps = ['dep'] * len(heads)
    tags = ['DT', 'NN', 'VBZ', '.']
    pos = ['DET', 'NOUN', 'VERB', 'PUNCT']
    doc = Doc(en_vocab, words=words, tags=tags, pos=pos, heads=heads, deps=deps, lemmas=lemmas)
    assert len(doc) == 4
    assert doc[0].text == 'The'
    assert doc[0].tag_ == 'DT'
    assert doc[0].pos_ == 'DET'
    assert doc[0].lemma_ == 'the'
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2])
    assert len(doc) == 3
    assert doc[0].text == 'The players'
    assert doc[0].tag_ == 'NN'
    assert doc[0].pos_ == 'NOUN'
    assert doc[0].lemma_ == 'the players'
    doc = Doc(en_vocab, words=words, tags=tags, pos=pos, heads=heads, deps=deps, lemmas=lemmas)
    assert len(doc) == 4
    assert doc[0].text == 'The'
    assert doc[0].tag_ == 'DT'
    assert doc[0].pos_ == 'DET'
    assert doc[0].lemma_ == 'the'
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2])
        retokenizer.merge(doc[2:4])
    assert len(doc) == 2
    assert doc[0].text == 'The players'
    assert doc[0].tag_ == 'NN'
    assert doc[0].pos_ == 'NOUN'
    assert doc[0].lemma_ == 'the players'
    assert doc[1].text == 'start .'
    assert doc[1].tag_ == 'VBZ'
    assert doc[1].pos_ == 'VERB'
    assert doc[1].lemma_ == 'start .'