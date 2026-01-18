import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_span_np_merges(en_tokenizer):
    text = 'displaCy is a parse tool built with Javascript'
    heads = [1, 1, 4, 4, 1, 4, 5, 6]
    deps = ['dep'] * len(heads)
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    assert doc[4].head.i == 1
    with doc.retokenize() as retokenizer:
        attrs = {'tag': 'NP', 'lemma': 'tool', 'ent_type': 'O'}
        retokenizer.merge(doc[2:5], attrs=attrs)
    assert doc[2].head.i == 1
    text = 'displaCy is a lightweight and modern dependency parse tree visualization tool built with CSS3 and JavaScript.'
    heads = [1, 1, 10, 7, 3, 3, 7, 10, 9, 10, 1, 10, 11, 12, 13, 13, 1]
    deps = ['dep'] * len(heads)
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            attrs = {'tag': ent.label_, 'lemma': ent.lemma_, 'ent_type': ent.label_}
            retokenizer.merge(ent, attrs=attrs)
    text = 'One test with entities like New York City so the ents list is not void'
    heads = [1, 1, 1, 2, 3, 6, 7, 4, 12, 11, 11, 12, 1, 12, 12]
    deps = ['dep'] * len(heads)
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)