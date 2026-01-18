from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magicmock_defaults(self):
    mock = MagicMock()
    self.assertEqual(int(mock), 1)
    self.assertEqual(complex(mock), 1j)
    self.assertEqual(float(mock), 1.0)
    self.assertEqual(long(mock), long(1))
    self.assertNotIn(object(), mock)
    self.assertEqual(len(mock), 0)
    self.assertEqual(list(mock), [])
    self.assertEqual(hash(mock), object.__hash__(mock))
    self.assertEqual(str(mock), object.__str__(mock))
    self.assertEqual(unicode(mock), object.__str__(mock))
    self.assertIsInstance(unicode(mock), unicode)
    self.assertTrue(bool(mock))
    if six.PY2:
        self.assertEqual(oct(mock), '1')
    else:
        self.assertEqual(oct(mock), '0o1')
    self.assertEqual(hex(mock), '0x1')