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
@pytest.mark.issue(6755)
def test_issue6755(en_tokenizer):
    doc = en_tokenizer('This is a magnificent sentence.')
    span = doc[:0]
    assert span.text_with_ws == ''
    assert span.text == ''