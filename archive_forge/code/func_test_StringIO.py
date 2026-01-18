import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_StringIO():
    fp = six.StringIO()
    fp.write(six.u('hello'))
    assert fp.getvalue() == six.u('hello')