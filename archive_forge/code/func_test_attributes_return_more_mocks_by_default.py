import unittest
from unittest import mock
def test_attributes_return_more_mocks_by_default(self):
    m = mock.Mock()
    self.assertIsInstance(m.test, mock.Mock)
    self.assertIsInstance(m.test(), mock.Mock)
    self.assertIsInstance(m.test().test2(), mock.Mock)