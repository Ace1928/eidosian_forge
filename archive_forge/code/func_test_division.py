from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_division(self):
    original = mock = Mock()
    mock.value = 32
    self.assertRaises(TypeError, lambda: mock / 2)

    def truediv(self, other):
        mock.value /= other
        return self
    mock.__truediv__ = truediv
    self.assertEqual(mock / 2, mock)
    self.assertEqual(mock.value, 16)
    del mock.__truediv__
    if six.PY3:

        def itruediv(mock):
            mock /= 4
        self.assertRaises(TypeError, itruediv, mock)
        mock.__itruediv__ = truediv
        mock /= 8
        self.assertEqual(mock, original)
        self.assertEqual(mock.value, 2)
    else:
        mock.value = 2
    self.assertRaises(TypeError, lambda: 8 / mock)
    mock.__rtruediv__ = truediv
    self.assertEqual(0.5 / mock, mock)
    self.assertEqual(mock.value, 4)