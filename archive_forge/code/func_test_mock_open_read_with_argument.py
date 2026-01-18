from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_mock_open_read_with_argument(self):
    some_data = 'foo\nbar\nbaz'
    mock = mock_open(read_data=some_data)
    self.assertEqual(mock().read(10), some_data)