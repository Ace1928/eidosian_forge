import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_ensure_text(self):
    converted_unicode = six.ensure_text(self.UNICODE_EMOJI, encoding='utf-8', errors='strict')
    converted_binary = six.ensure_text(self.BINARY_EMOJI, encoding='utf-8', errors='strict')
    if six.PY2:
        assert converted_unicode == self.UNICODE_EMOJI and isinstance(converted_unicode, unicode)
        assert converted_binary == self.UNICODE_EMOJI and isinstance(converted_unicode, unicode)
    else:
        assert converted_unicode == self.UNICODE_EMOJI and isinstance(converted_unicode, str)
        assert converted_binary == self.UNICODE_EMOJI and isinstance(converted_unicode, str)