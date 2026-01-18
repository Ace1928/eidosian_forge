from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_817234(self):
    it = regex.finditer('.*', 'asdf')
    self.assertEqual(next(it).span(), (0, 4))
    self.assertEqual(next(it).span(), (4, 4))
    self.assertRaises(StopIteration, lambda: next(it))