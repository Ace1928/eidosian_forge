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
def test_has_annotation_sents(en_vocab):
    doc = Doc(en_vocab, words=['Hello', 'beautiful', 'world'])
    attrs = ('SENT_START', 'IS_SENT_START', 'IS_SENT_END')
    for attr in attrs:
        assert not doc.has_annotation(attr)
        assert not doc.has_annotation(attr, require_complete=True)
    doc[1].is_sent_start = False
    for attr in attrs:
        assert doc.has_annotation(attr)
        assert not doc.has_annotation(attr, require_complete=True)
    doc[2].is_sent_start = False
    for attr in attrs:
        assert doc.has_annotation(attr)
        assert doc.has_annotation(attr, require_complete=True)