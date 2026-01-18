from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_449000(self):
    self.assertEqual(regex.sub('\\r\\n', '\\n', 'abc\r\ndef\r\n'), 'abc\ndef\n')
    self.assertEqual(regex.sub('\r\n', '\\n', 'abc\r\ndef\r\n'), 'abc\ndef\n')
    self.assertEqual(regex.sub('\\r\\n', '\n', 'abc\r\ndef\r\n'), 'abc\ndef\n')
    self.assertEqual(regex.sub('\r\n', '\n', 'abc\r\ndef\r\n'), 'abc\ndef\n')