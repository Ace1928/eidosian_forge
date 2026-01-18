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
def test_baseexceptional_side_effect(self):
    mock = Mock(side_effect=KeyboardInterrupt)
    self.assertRaises(KeyboardInterrupt, mock)
    mock = Mock(side_effect=KeyboardInterrupt('foo'))
    self.assertRaises(KeyboardInterrupt, mock)