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
@pytest.mark.parametrize('words,heads,lca_matrix', [(['the', 'lazy', 'dog', 'slept'], [2, 2, 3, 3], numpy.array([[0, 2, 2, 3], [2, 1, 2, 3], [2, 2, 2, 3], [3, 3, 3, 3]])), (['The', 'lazy', 'dog', 'slept', '.', 'The', 'quick', 'fox', 'jumped'], [2, 2, 3, 3, 3, 7, 7, 8, 8], numpy.array([[0, 2, 2, 3, 3, -1, -1, -1, -1], [2, 1, 2, 3, 3, -1, -1, -1, -1], [2, 2, 2, 3, 3, -1, -1, -1, -1], [3, 3, 3, 3, 3, -1, -1, -1, -1], [3, 3, 3, 3, 4, -1, -1, -1, -1], [-1, -1, -1, -1, -1, 5, 7, 7, 8], [-1, -1, -1, -1, -1, 7, 6, 7, 8], [-1, -1, -1, -1, -1, 7, 7, 7, 8], [-1, -1, -1, -1, -1, 8, 8, 8, 8]]))])
def test_lowest_common_ancestor(en_vocab, words, heads, lca_matrix):
    doc = Doc(en_vocab, words, heads=heads, deps=['dep'] * len(heads))
    lca = doc.get_lca_matrix()
    assert (lca == lca_matrix).all()
    assert lca[1, 1] == 1
    assert lca[0, 1] == 2
    assert lca[1, 2] == 2