import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_autospec_keywords(self):

    @patch('%s.function' % __name__, autospec=True, return_value=3)
    def test(mock_function):
        return function(1, 2)
    result = test()
    self.assertEqual(result, 3)