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
@pytest.mark.issue(2396)
def test_issue2396(en_vocab):
    words = ['She', 'created', 'a', 'test', 'for', 'spacy']
    heads = [1, 1, 3, 1, 3, 4]
    deps = ['dep'] * len(heads)
    matrix = numpy.array([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 2, 3, 3, 3], [1, 1, 3, 3, 3, 3], [1, 1, 3, 3, 4, 4], [1, 1, 3, 3, 4, 5]], dtype=numpy.int32)
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    span = doc[:]
    assert (doc.get_lca_matrix() == matrix).all()
    assert (span.get_lca_matrix() == matrix).all()