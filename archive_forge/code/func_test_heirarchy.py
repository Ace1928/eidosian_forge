import unittest2 as unittest
from mock.tests.support import is_instance, X, SomeClass
from mock import (
def test_heirarchy(self):
    self.assertTrue(issubclass(MagicMock, Mock))
    self.assertTrue(issubclass(NonCallableMagicMock, NonCallableMock))