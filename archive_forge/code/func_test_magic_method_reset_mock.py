from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magic_method_reset_mock(self):
    mock = MagicMock()
    str(mock)
    self.assertTrue(mock.__str__.called)
    mock.reset_mock()
    self.assertFalse(mock.__str__.called)