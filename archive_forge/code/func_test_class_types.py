import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_class_types():

    class X:
        pass

    class Y(object):
        pass
    assert isinstance(X, six.class_types)
    assert isinstance(Y, six.class_types)
    assert not isinstance(X(), six.class_types)