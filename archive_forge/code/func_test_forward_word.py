import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_forward_word(self):
    line = 'going from here to_here'
    start_pos = 11
    next_word_pos = 15
    expected = (next_word_pos, line)
    result = forward_word(start_pos, line)
    self.assertEqual(expected, result)
    start_pos = 15
    next_word_pos = 23
    expected = (next_word_pos, line)
    result = forward_word(start_pos, line)
    self.assertEqual(expected, result)