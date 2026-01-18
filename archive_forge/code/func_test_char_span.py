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
@pytest.mark.parametrize('i_sent,i,j,text', [(0, 0, len('This is a'), 'This is a'), (1, 0, len('This is another'), 'This is another'), (2, len('And '), len('And ') + len('a third'), 'a third'), (0, 1, 2, None)])
def test_char_span(doc, i_sent, i, j, text):
    sents = list(doc.sents)
    span = sents[i_sent].char_span(i, j)
    if not text:
        assert not span
    else:
        assert span.text == text