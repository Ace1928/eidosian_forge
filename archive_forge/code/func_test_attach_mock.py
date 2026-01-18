import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_attach_mock(self):
    classes = (Mock, MagicMock, NonCallableMagicMock, NonCallableMock)
    for Klass in classes:
        for Klass2 in classes:
            m = Klass()
            m2 = Klass2(name='foo')
            m.attach_mock(m2, 'bar')
            self.assertIs(m.bar, m2)
            self.assertIn("name='mock.bar'", repr(m2))
            m.bar.baz(1)
            self.assertEqual(m.mock_calls, [call.bar.baz(1)])
            self.assertEqual(m.method_calls, [call.bar.baz(1)])