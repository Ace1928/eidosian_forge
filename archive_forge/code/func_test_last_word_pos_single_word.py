import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_last_word_pos_single_word(self):
    line = 'word'
    expected = 0
    result = last_word_pos(line)
    self.assertEqual(expected, result)