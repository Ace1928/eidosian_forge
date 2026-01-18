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
def test_doc_ents_setter():
    """Test that both strings and integers can be used to set entities in
    tuple format via doc.ents."""
    words = ['a', 'b', 'c', 'd', 'e']
    doc = Doc(Vocab(), words=words)
    doc.ents = [('HELLO', 0, 2), (doc.vocab.strings.add('WORLD'), 3, 5)]
    assert [e.label_ for e in doc.ents] == ['HELLO', 'WORLD']
    vocab = Vocab()
    ents = [('HELLO', 0, 2), (vocab.strings.add('WORLD'), 3, 5)]
    ents = ['B-HELLO', 'I-HELLO', 'O', 'B-WORLD', 'I-WORLD']
    doc = Doc(vocab, words=words, ents=ents)
    assert [e.label_ for e in doc.ents] == ['HELLO', 'WORLD']