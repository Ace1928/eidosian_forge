import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_create_unbound_method():

    class X(object):
        pass

    def f(self):
        return self
    u = six.create_unbound_method(f, X)
    pytest.raises(TypeError, u)
    if six.PY2:
        assert isinstance(u, types.MethodType)
    x = X()
    assert f(x) is x