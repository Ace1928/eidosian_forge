from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_418626(self):
    self.assertEqual(regex.match('.*?c', 10000 * 'ab' + 'cd').end(0), 20001)
    self.assertEqual(regex.match('.*?cd', 5000 * 'ab' + 'c' + 5000 * 'ab' + 'cde').end(0), 20003)
    self.assertEqual(regex.match('.*?cd', 20000 * 'abc' + 'de').end(0), 60001)
    self.assertEqual(regex.search('(a|b)*?c', 10000 * 'ab' + 'cd').end(0), 20001)