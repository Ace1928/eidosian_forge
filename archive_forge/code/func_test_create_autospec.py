import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_create_autospec(self):
    mock = create_autospec(X)
    instance = mock()
    self.assertRaises(TypeError, instance)
    mock = create_autospec(X())
    self.assertRaises(TypeError, mock)