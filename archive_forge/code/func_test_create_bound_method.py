import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_create_bound_method():

    class X(object):
        pass

    def f(self):
        return self
    x = X()
    b = six.create_bound_method(f, x)
    assert isinstance(b, types.MethodType)
    assert b() is x