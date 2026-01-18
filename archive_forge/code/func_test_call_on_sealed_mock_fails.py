import unittest
from unittest import mock
def test_call_on_sealed_mock_fails(self):
    m = mock.Mock()
    mock.seal(m)
    with self.assertRaises(AttributeError):
        m()