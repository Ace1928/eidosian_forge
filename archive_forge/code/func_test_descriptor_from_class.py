from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_descriptor_from_class(self):
    m = MagicMock()
    type(m).__str__.return_value = 'foo'
    self.assertEqual(str(m), 'foo')