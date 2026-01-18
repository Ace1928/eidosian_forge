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
@pytest.mark.parametrize('sentence', ['The story was to the effect that a young American student recently called on Professor Christlieb with a letter of introduction.', "The next month Barry Siddall joined Stoke City on a free transfer, after Chris Pearce had established himself as the Vale's #1.", "The next month Barry Siddall joined Stoke City on a free transfer, after Chris Pearce had established himself as the Vale's number one", 'Indeed, making the one who remains do all the work has installed him into a position of such insolent tyranny, it will take a month at least to reduce him to his proper proportions.', "It was a missed assignment, but it shouldn't have resulted in a turnover ..."])
@pytest.mark.issue(3869)
def test_issue3869(sentence):
    """Test that the Doc's count_by function works consistently"""
    nlp = English()
    doc = nlp(sentence)
    count = 0
    for token in doc:
        count += token.is_alpha
    assert count == doc.count_by(IS_ALPHA).get(1, 0)