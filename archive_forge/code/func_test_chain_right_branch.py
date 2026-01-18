import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_chain_right_branch(model1, model2, model3):
    merge1 = chain(model1, model2)
    merge2 = chain(merge1, model3)
    assert len(merge1.layers) == 2
    assert len(merge2.layers) == 2