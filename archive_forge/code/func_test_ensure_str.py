import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_ensure_str(self):
    converted_unicode = six.ensure_str(self.UNICODE_EMOJI, encoding='utf-8', errors='strict')
    converted_binary = six.ensure_str(self.BINARY_EMOJI, encoding='utf-8', errors='strict')
    if six.PY2:
        assert converted_unicode == self.BINARY_EMOJI and isinstance(converted_unicode, str)
        assert converted_binary == self.BINARY_EMOJI and isinstance(converted_binary, str)
    else:
        assert converted_unicode == self.UNICODE_EMOJI and isinstance(converted_unicode, str)
        assert converted_binary == self.UNICODE_EMOJI and isinstance(converted_unicode, str)