import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_subtree_size_check(en_tokenizer):
    text = 'Stewart Lee is a stand up comedian who lives in England and loves Joe Pasquale'
    heads = [1, 2, 2, 4, 6, 4, 2, 8, 6, 8, 9, 8, 8, 14, 12]
    deps = ['compound', 'nsubj', 'ROOT', 'det', 'amod', 'prt', 'attr', 'nsubj', 'relcl', 'prep', 'pobj', 'cc', 'conj', 'compound', 'dobj']
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps)
    sent1 = list(doc.sents)[0]
    init_len = len(list(sent1.root.subtree))
    with doc.retokenize() as retokenizer:
        attrs = {'lemma': 'none', 'ent_type': 'none'}
        retokenizer.merge(doc[0:2], attrs=attrs)
    assert len(list(sent1.root.subtree)) == init_len - 1