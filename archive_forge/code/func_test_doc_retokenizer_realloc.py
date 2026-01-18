import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenizer_realloc(en_vocab):
    """#4604: realloc correctly when new tokens outnumber original tokens"""
    text = 'Hyperglycemic adverse events following antipsychotic drug administration in the'
    doc = Doc(en_vocab, words=text.split()[:-1])
    with doc.retokenize() as retokenizer:
        token = doc[0]
        heads = [(token, 0)] * len(token)
        retokenizer.split(doc[token.i], list(token.text), heads=heads)
    doc = Doc(en_vocab, words=text.split())
    with doc.retokenize() as retokenizer:
        token = doc[0]
        heads = [(token, 0)] * len(token)
        retokenizer.split(doc[token.i], list(token.text), heads=heads)