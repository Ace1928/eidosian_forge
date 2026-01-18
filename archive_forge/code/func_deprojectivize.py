import pytest
from spacy.pipeline._parser_internals import nonproj
from spacy.pipeline._parser_internals.nonproj import (
from spacy.tokens import Doc
def deprojectivize(proj_heads, deco_labels):
    words = ['whatever '] * len(proj_heads)
    doc = Doc(en_vocab, words=words, deps=deco_labels, heads=proj_heads)
    nonproj.deprojectivize(doc)
    return ([t.head.i for t in doc], [token.dep_ for token in doc])