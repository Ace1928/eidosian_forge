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
def get_gradient(model, Y):
    if isinstance(Y, model.ops.xp.ndarray):
        dY = model.ops.alloc(Y.shape, dtype=Y.dtype)
        dY += model.ops.xp.random.uniform(-1.0, 1.0, Y.shape)
        return dY
    elif isinstance(Y, List):
        return [get_gradient(model, y) for y in Y]
    else:
        raise ValueError(f'Could not get gradient for type {type(Y)}')