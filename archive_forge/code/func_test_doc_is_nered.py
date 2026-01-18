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
def test_doc_is_nered(en_vocab):
    words = ['I', 'live', 'in', 'New', 'York']
    doc = Doc(en_vocab, words=words)
    assert not doc.has_annotation('ENT_IOB')
    doc.ents = [Span(doc, 3, 5, label='GPE')]
    assert doc.has_annotation('ENT_IOB')
    arr = numpy.array([[0, 0], [0, 0], [0, 0], [384, 3], [384, 1]], dtype='uint64')
    doc = Doc(en_vocab, words=words).from_array([ENT_TYPE, ENT_IOB], arr)
    assert doc.has_annotation('ENT_IOB')
    new_doc = Doc(en_vocab).from_bytes(doc.to_bytes())
    assert new_doc.has_annotation('ENT_IOB')