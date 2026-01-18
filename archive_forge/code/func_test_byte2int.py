import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_byte2int():
    assert six.byte2int(six.b('\x03')) == 3
    assert six.byte2int(six.b('\x03\x04')) == 3
    pytest.raises(IndexError, six.byte2int, six.b(''))