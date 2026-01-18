import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_functions_with_bad_signatures(self):

    def f(something):
        return (1, 2)
    with self.assertRaises(TypeError):
        self.edits.add('a', f)

    def g(cursor_offset, line, something, something_else):
        return (1, 2)
    with self.assertRaises(TypeError):
        self.edits.add('a', g)