import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
@pytest.mark.parametrize('i', [slice(5), slice(5, 0), asarray(True), asarray([0, 1])])
def test_error_on_invalid_index_with_ellipsis(i):
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[..., i]
    with pytest.raises(IndexError):
        a[i, ...]