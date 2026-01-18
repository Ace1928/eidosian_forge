import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_get_function_code():

    def f():
        pass
    assert isinstance(six.get_function_code(f), types.CodeType)
    if not hasattr(sys, 'pypy_version_info'):
        pytest.raises(AttributeError, six.get_function_code, hasattr)