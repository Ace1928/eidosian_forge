from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_context_manager_with_magic_mock(self):
    mock = MagicMock()
    with self.assertRaises(TypeError):
        with mock:
            'foo' + 3
    mock.__enter__.assert_called_with()
    self.assertTrue(mock.__exit__.called)