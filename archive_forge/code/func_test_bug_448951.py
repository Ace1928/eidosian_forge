from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_448951(self):
    for op in ('', '?', '*'):
        self.assertEqual(regex.match('((.%s):)?z' % op, 'z')[:], ('z', None, None))
        self.assertEqual(regex.match('((.%s):)?z' % op, 'a:z')[:], ('a:z', 'a:', 'a'))