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
def test_many_duck_arrays(self):

    class A:
        __array_function__ = _return_not_implemented

    class B(A):
        __array_function__ = _return_not_implemented

    class C(A):
        __array_function__ = _return_not_implemented

    class D:
        __array_function__ = _return_not_implemented
    a = A()
    b = B()
    c = C()
    d = D()
    assert_equal(_get_implementing_args([1]), [])
    assert_equal(_get_implementing_args([a]), [a])
    assert_equal(_get_implementing_args([a, 1]), [a])
    assert_equal(_get_implementing_args([a, a, a]), [a])
    assert_equal(_get_implementing_args([a, d, a]), [a, d])
    assert_equal(_get_implementing_args([a, b]), [b, a])
    assert_equal(_get_implementing_args([b, a]), [b, a])
    assert_equal(_get_implementing_args([a, b, c]), [b, c, a])
    assert_equal(_get_implementing_args([a, c, b]), [c, b, a])