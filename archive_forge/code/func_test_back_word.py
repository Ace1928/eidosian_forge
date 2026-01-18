import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_back_word(self):
    line = 'going to here from_here'
    start_pos = 14
    prev_word_pos = 9
    self.assertEqual(line[start_pos], 'f')
    self.assertEqual(line[prev_word_pos], 'h')
    expected = (prev_word_pos, line)
    result = back_word(start_pos, line)
    self.assertEqual(expected, result)