import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import get_current_ops
from spacy.attrs import LENGTH, ORTH
from spacy.lang.en import English
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab
from .test_underscore import clean_underscore  # noqa: F401
@pytest.mark.issue(1612)
def test_issue1612(en_tokenizer):
    """Test that span.orth_ is identical to span.text"""
    doc = en_tokenizer('The black cat purrs.')
    span = doc[1:3]
    assert span.orth_ == span.text