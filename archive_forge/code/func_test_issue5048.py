import warnings
import weakref
import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.attrs import (
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from spacy.lexeme import Lexeme
from spacy.tokens import Doc, Span, SpanGroup, Token
from spacy.vocab import Vocab
from .test_underscore import clean_underscore  # noqa: F401
@pytest.mark.issue(5048)
def test_issue5048(en_vocab):
    words = ['This', 'is', 'a', 'sentence']
    pos_s = ['DET', 'VERB', 'DET', 'NOUN']
    spaces = [' ', ' ', ' ', '']
    deps_s = ['dep', 'adj', 'nn', 'atm']
    tags_s = ['DT', 'VBZ', 'DT', 'NN']
    strings = en_vocab.strings
    for w in words:
        strings.add(w)
    deps = [strings.add(d) for d in deps_s]
    pos = [strings.add(p) for p in pos_s]
    tags = [strings.add(t) for t in tags_s]
    attrs = [POS, DEP, TAG]
    array = numpy.array(list(zip(pos, deps, tags)), dtype='uint64')
    doc = Doc(en_vocab, words=words, spaces=spaces)
    doc.from_array(attrs, array)
    v1 = [(token.text, token.pos_, token.tag_) for token in doc]
    doc2 = Doc(en_vocab, words=words, pos=pos_s, deps=deps_s, tags=tags_s)
    v2 = [(token.text, token.pos_, token.tag_) for token in doc2]
    assert v1 == v2