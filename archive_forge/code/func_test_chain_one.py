import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_chain_one(model1):
    with pytest.raises(TypeError):
        chain(model1)