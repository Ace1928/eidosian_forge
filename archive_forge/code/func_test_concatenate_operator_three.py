import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_concatenate_operator_three(model1, model2, model3):
    with Model.define_operators({'|': concatenate}):
        model = model1 | model2 | model3
        assert len(model.layers) == 3