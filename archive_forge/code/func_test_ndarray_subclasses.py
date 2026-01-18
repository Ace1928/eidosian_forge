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
def test_ndarray_subclasses(self):

    class OverrideSub(np.ndarray):
        __array_function__ = _return_not_implemented

    class NoOverrideSub(np.ndarray):
        pass
    array = np.array(1).view(np.ndarray)
    override_sub = np.array(1).view(OverrideSub)
    no_override_sub = np.array(1).view(NoOverrideSub)
    args = _get_implementing_args([array, override_sub])
    assert_equal(list(args), [override_sub, array])
    args = _get_implementing_args([array, no_override_sub])
    assert_equal(list(args), [no_override_sub, array])
    args = _get_implementing_args([override_sub, no_override_sub])
    assert_equal(list(args), [override_sub, no_override_sub])