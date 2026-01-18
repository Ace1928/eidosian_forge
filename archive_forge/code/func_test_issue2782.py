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
@pytest.mark.parametrize('text', ['-0.23', '+123,456', '±1'])
@pytest.mark.parametrize('lang_cls', [English, MultiLanguage])
@pytest.mark.issue(2782)
def test_issue2782(text, lang_cls):
    """Check that like_num handles + and - before number."""
    nlp = lang_cls()
    doc = nlp(text)
    assert len(doc) == 1
    assert doc[0].like_num