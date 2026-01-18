import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_get_method_function():

    class X(object):

        def m(self):
            pass
    x = X()
    assert six.get_method_function(x.m) is X.__dict__['m']
    pytest.raises(AttributeError, six.get_method_function, hasattr)