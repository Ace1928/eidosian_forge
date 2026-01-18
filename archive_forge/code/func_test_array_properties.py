import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_array_properties():
    a = ones((1, 2, 3))
    b = ones((2, 3))
    assert_raises(ValueError, lambda: a.T)
    assert isinstance(b.T, Array)
    assert b.T.shape == (3, 2)
    assert isinstance(a.mT, Array)
    assert a.mT.shape == (1, 3, 2)
    assert isinstance(b.mT, Array)
    assert b.mT.shape == (3, 2)