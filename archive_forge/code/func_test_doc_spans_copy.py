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
def test_doc_spans_copy(en_tokenizer):
    doc1 = en_tokenizer('Some text about Colombia and the Czech Republic')
    assert weakref.ref(doc1) == doc1.spans.doc_ref
    doc2 = doc1.copy()
    assert weakref.ref(doc2) == doc2.spans.doc_ref