import unittest
from unittest import mock
def test_new_attributes_cannot_be_accessed_on_seal(self):
    m = mock.Mock()
    mock.seal(m)
    with self.assertRaises(AttributeError):
        m.test
    with self.assertRaises(AttributeError):
        m()