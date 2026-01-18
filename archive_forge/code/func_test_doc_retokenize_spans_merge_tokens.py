import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_merge_tokens(en_tokenizer):
    text = 'Los Angeles start.'
    heads = [1, 2, 2, 2]
    deps = ['dep'] * len(heads)
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    assert len(doc) == 4
    assert doc[0].head.text == 'Angeles'
    assert doc[1].head.text == 'start'
    with doc.retokenize() as retokenizer:
        attrs = {'tag': 'NNP', 'lemma': 'Los Angeles', 'ent_type': 'GPE'}
        retokenizer.merge(doc[0:2], attrs=attrs)
    assert len(doc) == 3
    assert doc[0].text == 'Los Angeles'
    assert doc[0].head.text == 'start'
    assert doc[0].ent_type_ == 'GPE'