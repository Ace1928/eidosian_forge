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
def test_doc_from_array_sent_starts(en_vocab):
    words = ['I', 'live', 'in', 'New', 'York', '.', 'I', 'like', 'cats', '.']
    heads = [0, 0, 0, 0, 0, 0, 6, 6, 6, 6]
    deps = ['ROOT', 'dep', 'dep', 'dep', 'dep', 'dep', 'ROOT', 'dep', 'dep', 'dep']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    attrs = [SENT_START, HEAD]
    arr = doc.to_array(attrs)
    new_doc = Doc(en_vocab, words=words)
    new_doc.from_array(attrs, arr)
    attrs = doc._get_array_attrs()
    arr = doc.to_array(attrs)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        new_doc.from_array(attrs, arr)
    attrs = [SENT_START]
    arr = doc.to_array(attrs)
    new_doc = Doc(en_vocab, words=words)
    new_doc.from_array(attrs, arr)
    assert [t.is_sent_start for t in doc] == [t.is_sent_start for t in new_doc]
    assert not new_doc.has_annotation('DEP')
    attrs = [HEAD, DEP]
    arr = doc.to_array(attrs)
    new_doc = Doc(en_vocab, words=words)
    new_doc.from_array(attrs, arr)
    assert [t.is_sent_start for t in doc] == [t.is_sent_start for t in new_doc]
    assert new_doc.has_annotation('DEP')