from typing import List
import numpy
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from thinc.api import (
from spacy.lang.en import English
from spacy.lang.en.examples import sentences as EN_SENTENCES
from spacy.ml.extract_spans import _get_span_indices, extract_spans
from spacy.ml.models import (
from spacy.ml.staticvectors import StaticVectors
from spacy.util import registry
def test_extract_spans_forward_backward():
    model = extract_spans().initialize()
    X = Ragged(model.ops.alloc2f(15, 4), model.ops.asarray([5, 10], dtype='i'))
    spans = Ragged(model.ops.asarray([[0, 3], [2, 3], [5, 7]], dtype='i'), model.ops.asarray([2, 1], dtype='i'))
    Y, backprop = model.begin_update((X, spans))
    assert list(Y.lengths) == [3, 1, 2]
    assert Y.dataXd.shape == (6, 4)
    dX, spans2 = backprop(Y)
    assert spans2 is spans
    assert dX.dataXd.shape == X.dataXd.shape
    assert list(dX.lengths) == list(X.lengths)