import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_autospec_function(self):

    @patch('%s.function' % __name__, autospec=True)
    def test(mock):
        function(1)
        function.assert_called_with(1)
        function(2, 3)
        function.assert_called_with(2, 3)
        self.assertRaises(TypeError, function)
        self.assertRaises(AttributeError, getattr, function, 'foo')
    test()