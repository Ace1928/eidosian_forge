import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_allow_newaxis():
    a = ones(5)
    indexed_a = a[None, :]
    assert indexed_a.shape == (1, 5)