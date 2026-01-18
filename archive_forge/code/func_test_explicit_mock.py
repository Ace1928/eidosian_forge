from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_explicit_mock(self):
    mock = MagicMock()
    mock_open(mock)
    with patch('%s.open' % __name__, mock, create=True) as patched:
        self.assertIs(patched, mock)
        open('foo')
    mock.assert_called_once_with('foo')