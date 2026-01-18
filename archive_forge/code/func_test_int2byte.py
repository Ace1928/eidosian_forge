import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_int2byte():
    assert six.int2byte(3) == six.b('\x03')
    pytest.raises(Exception, six.int2byte, 256)