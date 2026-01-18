import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_chain_three(model1, model2, model3):
    model = chain(model1, model2, model3)
    assert len(model.layers) == 3