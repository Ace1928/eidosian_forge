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
def test_magic_methods_mock_calls(self):
    for Klass in (Mock, MagicMock):
        m = Klass()
        m.__int__ = Mock(return_value=3)
        m.__float__ = MagicMock(return_value=3.0)
        int(m)
        float(m)
        self.assertEqual(m.mock_calls, [call.__int__(), call.__float__()])
        self.assertEqual(m.method_calls, [])