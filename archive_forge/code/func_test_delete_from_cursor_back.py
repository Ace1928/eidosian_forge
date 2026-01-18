import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_delete_from_cursor_back(self):
    line = 'everything before this will be deleted'
    expected = (0, 'this will be deleted')
    result = delete_from_cursor_back(line.find('this'), line)
    self.assertEqual(expected, result)