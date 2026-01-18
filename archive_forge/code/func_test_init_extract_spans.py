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
def test_init_extract_spans():
    extract_spans().initialize()