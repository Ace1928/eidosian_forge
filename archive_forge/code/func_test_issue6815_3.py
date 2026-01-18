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
@pytest.mark.parametrize('sentence, start_idx,end_idx,vector', [('Welcome to Mumbai, my friend', 11, 17, numpy.array([0.1, 0.2, 0.3]))])
@pytest.mark.issue(6815)
def test_issue6815_3(sentence, start_idx, end_idx, vector):
    nlp = English()
    doc = nlp(sentence)
    span = doc[:].char_span(start_idx, end_idx, vector=vector)
    assert (span.vector == vector).all()