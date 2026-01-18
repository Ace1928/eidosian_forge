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
def test_filter_spans(doc):
    spans = [doc[1:4], doc[6:8], doc[1:4], doc[10:14]]
    filtered = filter_spans(spans)
    assert len(filtered) == 3
    assert filtered[0].start == 1 and filtered[0].end == 4
    assert filtered[1].start == 6 and filtered[1].end == 8
    assert filtered[2].start == 10 and filtered[2].end == 14
    spans = [doc[1:4], doc[1:3], doc[5:10], doc[7:9], doc[1:4]]
    filtered = filter_spans(spans)
    assert len(filtered) == 2
    assert len(filtered[0]) == 3
    assert len(filtered[1]) == 5
    assert filtered[0].start == 1 and filtered[0].end == 4
    assert filtered[1].start == 5 and filtered[1].end == 10
    spans = [doc[1:4], doc[2:5], doc[5:10], doc[7:9], doc[1:4]]
    filtered = filter_spans(spans)
    assert len(filtered) == 2
    assert len(filtered[0]) == 3
    assert len(filtered[1]) == 5
    assert filtered[0].start == 1 and filtered[0].end == 4
    assert filtered[1].start == 5 and filtered[1].end == 10