import unittest
from unittest import mock
def test_initialized_attributes_allowed_after_seal(self):
    m = mock.Mock(test_value=1)
    mock.seal(m)
    self.assertEqual(m.test_value, 1)