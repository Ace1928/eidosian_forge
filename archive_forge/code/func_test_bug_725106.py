from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_725106(self):
    self.assertEqual(regex.match('^((a)|b)*', 'abc')[:], ('ab', 'b', 'a'))
    self.assertEqual(regex.match('^(([ab])|c)*', 'abc')[:], ('abc', 'c', 'b'))
    self.assertEqual(regex.match('^((d)|[ab])*', 'abc')[:], ('ab', 'b', None))
    self.assertEqual(regex.match('^((a)c|[ab])*', 'abc')[:], ('ab', 'b', None))
    self.assertEqual(regex.match('^((a)|b)*?c', 'abc')[:], ('abc', 'b', 'a'))
    self.assertEqual(regex.match('^(([ab])|c)*?d', 'abcd')[:], ('abcd', 'c', 'b'))
    self.assertEqual(regex.match('^((d)|[ab])*?c', 'abc')[:], ('abc', 'b', None))
    self.assertEqual(regex.match('^((a)c|[ab])*?c', 'abc')[:], ('abc', 'b', None))