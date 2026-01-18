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
def test_doc_api_init(en_vocab):
    words = ['a', 'b', 'c', 'd']
    heads = [0, 0, 2, 2]
    doc = Doc(en_vocab, words=words, sent_starts=[True, False, True, False])
    assert [t.is_sent_start for t in doc] == [True, False, True, False]
    doc = Doc(en_vocab, words=words, heads=heads, deps=['dep'] * 4)
    assert [t.is_sent_start for t in doc] == [True, False, True, False]
    doc = Doc(en_vocab, words=words, sent_starts=[True] * 4, heads=heads, deps=['dep'] * 4)
    assert [t.is_sent_start for t in doc] == [True, False, True, False]