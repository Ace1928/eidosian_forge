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
def test_doc_spans_setdefault(en_tokenizer):
    doc = en_tokenizer('Some text about Colombia and the Czech Republic')
    doc.spans.setdefault('key1')
    assert len(doc.spans['key1']) == 0
    doc.spans.setdefault('key2', default=[doc[0:1]])
    assert len(doc.spans['key2']) == 1
    doc.spans.setdefault('key3', default=SpanGroup(doc, spans=[doc[0:1], doc[1:2]]))
    assert len(doc.spans['key3']) == 2