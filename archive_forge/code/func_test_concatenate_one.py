import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_concatenate_one(model1):
    model = concatenate(model1)
    assert isinstance(model, Model)