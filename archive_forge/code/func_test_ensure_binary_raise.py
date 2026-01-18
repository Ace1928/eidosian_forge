import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_ensure_binary_raise(self):
    converted_unicode = six.ensure_binary(self.UNICODE_EMOJI, encoding='utf-8', errors='strict')
    converted_binary = six.ensure_binary(self.BINARY_EMOJI, encoding='utf-8', errors='strict')
    if six.PY2:
        assert converted_unicode == self.BINARY_EMOJI and isinstance(converted_unicode, str)
        assert converted_binary == self.BINARY_EMOJI and isinstance(converted_binary, str)
    else:
        assert converted_unicode == self.BINARY_EMOJI and isinstance(converted_unicode, bytes)
        assert converted_binary == self.BINARY_EMOJI and isinstance(converted_binary, bytes)