from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_113254(self):
    self.assertEqual(regex.match('(a)|(b)', 'b').start(1), -1)
    self.assertEqual(regex.match('(a)|(b)', 'b').end(1), -1)
    self.assertEqual(regex.match('(a)|(b)', 'b').span(1), (-1, -1))