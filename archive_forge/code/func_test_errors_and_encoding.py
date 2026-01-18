import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_errors_and_encoding(self):
    six.ensure_binary(self.UNICODE_EMOJI, encoding='latin-1', errors='ignore')
    with pytest.raises(UnicodeEncodeError):
        six.ensure_binary(self.UNICODE_EMOJI, encoding='latin-1', errors='strict')