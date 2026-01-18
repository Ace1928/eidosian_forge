from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_dict_context_manager(self):
    foo = {}
    with patch.dict(foo, {'a': 'b'}):
        self.assertEqual(foo, {'a': 'b'})
    self.assertEqual(foo, {})
    with self.assertRaises(NameError):
        with patch.dict(foo, {'a': 'b'}):
            self.assertEqual(foo, {'a': 'b'})
            raise NameError('Konrad')
    self.assertEqual(foo, {})