from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_dollar_matches_twice(self):
    pattern = regex.compile('$')
    self.assertEqual(pattern.sub('#', 'a\nb\n'), 'a\nb#\n#')
    self.assertEqual(pattern.sub('#', 'a\nb\nc'), 'a\nb\nc#')
    self.assertEqual(pattern.sub('#', '\n'), '#\n#')
    pattern = regex.compile('$', regex.MULTILINE)
    self.assertEqual(pattern.sub('#', 'a\nb\n'), 'a#\nb#\n#')
    self.assertEqual(pattern.sub('#', 'a\nb\nc'), 'a#\nb#\nc#')
    self.assertEqual(pattern.sub('#', '\n'), '#\n#')