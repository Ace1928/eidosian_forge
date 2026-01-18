import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_forward_word_empty(self):
    line = ''
    start_pos = 0
    next_word_pos = 0
    expected = (next_word_pos, line)
    result = forward_word(start_pos, line)
    self.assertEqual(expected, result)