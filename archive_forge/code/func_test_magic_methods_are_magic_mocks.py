from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_methods_are_magic_mocks(self):
    mock = MagicMock()
    self.assertIsInstance(mock.__getitem__, MagicMock)
    mock[1][2].__getitem__.return_value = 3
    self.assertEqual(mock[1][2][3], 3)