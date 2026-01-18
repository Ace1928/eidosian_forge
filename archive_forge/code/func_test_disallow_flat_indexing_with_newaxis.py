import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_disallow_flat_indexing_with_newaxis():
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[None, 0, 0]