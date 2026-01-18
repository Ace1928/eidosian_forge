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
def test_create_autospec_with_name(self):
    m = mock.create_autospec(object(), name='sweet_func')
    self.assertIn('sweet_func', repr(m))