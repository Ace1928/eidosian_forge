import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_beginning_of_line(self):
    expected = (0, self._line)
    for i in range(len(self._line)):
        result = beginning_of_line(i, self._line)
        self.assertEqual(expected, result)