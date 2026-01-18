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
def get_updated_model():
    fix_random_seed(seed)
    optimizer = Adam(0.001)
    model = model_func(**kwargs).initialize()
    initial_params = get_all_params(model)
    set_dropout_rate(model, dropout)
    for _ in range(5):
        Y, get_dX = model.begin_update(get_X())
        dY = get_gradient(model, Y)
        get_dX(dY)
        model.finish_update(optimizer)
    updated_params = get_all_params(model)
    with pytest.raises(AssertionError):
        assert_array_equal(model.ops.to_numpy(initial_params), model.ops.to_numpy(updated_params))
    return model