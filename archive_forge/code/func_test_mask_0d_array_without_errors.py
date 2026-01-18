import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_mask_0d_array_without_errors():
    a = ones(())
    a[asarray(True)]