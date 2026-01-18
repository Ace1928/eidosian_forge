import unittest
from unittest import mock
def test_call_on_defined_sealed_mock_succeeds(self):
    m = mock.Mock(return_value=5)
    mock.seal(m)
    self.assertEqual(m(), 5)