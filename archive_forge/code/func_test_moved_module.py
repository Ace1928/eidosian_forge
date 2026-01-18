import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_moved_module(self):
    attr = six.MovedModule('spam', 'foo')
    if six.PY3:
        assert attr.mod == 'spam'
    else:
        assert attr.mod == 'foo'
    attr = six.MovedModule('spam', 'foo', 'bar')
    if six.PY3:
        assert attr.mod == 'bar'
    else:
        assert attr.mod == 'foo'