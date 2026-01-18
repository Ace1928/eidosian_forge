from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_1661(self):
    pattern = regex.compile('.')
    self.assertRaisesRegex(ValueError, self.FLAGS_WITH_COMPILED_PAT, lambda: regex.match(pattern, 'A', regex.I))
    self.assertRaisesRegex(ValueError, self.FLAGS_WITH_COMPILED_PAT, lambda: regex.search(pattern, 'A', regex.I))
    self.assertRaisesRegex(ValueError, self.FLAGS_WITH_COMPILED_PAT, lambda: regex.findall(pattern, 'A', regex.I))
    self.assertRaisesRegex(ValueError, self.FLAGS_WITH_COMPILED_PAT, lambda: regex.compile(pattern, regex.I))