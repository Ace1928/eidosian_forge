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
def test_from_spec(self):

    class Something(object):
        x = 3
        __something__ = None

        def y(self):
            pass

    def test_attributes(mock):
        mock.x
        mock.y
        mock.__something__
        self.assertRaisesRegex(AttributeError, "Mock object has no attribute 'z'", getattr, mock, 'z')
        self.assertRaisesRegex(AttributeError, "Mock object has no attribute '__foobar__'", getattr, mock, '__foobar__')
    test_attributes(Mock(spec=Something))
    test_attributes(Mock(spec=Something()))