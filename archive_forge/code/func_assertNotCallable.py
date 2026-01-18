import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def assertNotCallable(self, mock):
    self.assertTrue(is_instance(mock, NonCallableMagicMock))
    self.assertFalse(is_instance(mock, CallableMixin))