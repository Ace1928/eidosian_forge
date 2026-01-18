import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_functions_with_bad_return_values(self):

    def f(cursor_offset, line):
        return ('hi',)
    with self.assertRaises(ValueError):
        self.edits.add('a', f)

    def g(cursor_offset, line):
        return ('hi', 1, 2, 3)
    with self.assertRaises(ValueError):
        self.edits.add('b', g)