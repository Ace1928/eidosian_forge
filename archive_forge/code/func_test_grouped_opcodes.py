import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_grouped_opcodes(self):

    def chk_ops(a, b, expected_codes, n=3):
        s = self._PatienceSequenceMatcher(None, a, b)
        self.assertEqual(expected_codes, list(s.get_grouped_opcodes(n)))
    chk_ops('', '', [])
    chk_ops([], [], [])
    chk_ops('abc', '', [[('delete', 0, 3, 0, 0)]])
    chk_ops('', 'abc', [[('insert', 0, 0, 0, 3)]])
    chk_ops('abcd', 'abcd', [])
    chk_ops('abcd', 'abce', [[('equal', 0, 3, 0, 3), ('replace', 3, 4, 3, 4)]])
    chk_ops('eabc', 'abce', [[('delete', 0, 1, 0, 0), ('equal', 1, 4, 0, 3), ('insert', 4, 4, 3, 4)]])
    chk_ops('abcdefghijklmnop', 'abcdefxydefghijklmnop', [[('equal', 3, 6, 3, 6), ('insert', 6, 6, 6, 11), ('equal', 6, 9, 11, 14)]])
    chk_ops('abcdefghijklmnop', 'abcdefxydefghijklmnop', [[('equal', 2, 6, 2, 6), ('insert', 6, 6, 6, 11), ('equal', 6, 10, 11, 15)]], 4)
    chk_ops('Xabcdef', 'abcdef', [[('delete', 0, 1, 0, 0), ('equal', 1, 4, 0, 3)]])
    chk_ops('abcdef', 'abcdefX', [[('equal', 3, 6, 3, 6), ('insert', 6, 6, 6, 7)]])