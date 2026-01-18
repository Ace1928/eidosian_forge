import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_ndarray_subclass_and_duck_array(self):

    class OverrideSub(np.ndarray):
        __array_function__ = _return_not_implemented

    class Other:
        __array_function__ = _return_not_implemented
    array = np.array(1)
    subarray = np.array(1).view(OverrideSub)
    other = Other()
    assert_equal(_get_implementing_args([array, subarray, other]), [subarray, array, other])
    assert_equal(_get_implementing_args([array, other, subarray]), [subarray, array, other])