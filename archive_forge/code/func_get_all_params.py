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
def get_all_params(model):
    params = []
    for node in model.walk():
        for name in node.param_names:
            params.append(node.get_param(name).ravel())
    return node.ops.xp.concatenate(params)