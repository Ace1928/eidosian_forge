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
def test_exceptional_side_effect(self):
    mock = Mock(side_effect=AttributeError)
    self.assertRaises(AttributeError, mock)
    mock = Mock(side_effect=AttributeError('foo'))
    self.assertRaises(AttributeError, mock)