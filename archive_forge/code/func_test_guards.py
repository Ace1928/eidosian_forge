from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_guards(self):
    m = regex.search('(X.*?Y\\s*){3}(X\\s*)+AB:', 'XY\nX Y\nX  Y\nXY\nXX AB:')
    self.assertEqual(m.span(0, 1, 2), ((3, 21), (12, 15), (16, 18)))
    m = regex.search('(X.*?Y\\s*){3,}(X\\s*)+AB:', 'XY\nX Y\nX  Y\nXY\nXX AB:')
    self.assertEqual(m.span(0, 1, 2), ((0, 21), (12, 15), (16, 18)))
    m = regex.search('\\d{4}(\\s*\\w)?\\W*((?!\\d)\\w){2}', '9999XX')
    self.assertEqual(m.span(0, 1, 2), ((0, 6), (-1, -1), (5, 6)))
    m = regex.search('A\\s*?.*?(\\n+.*?\\s*?){0,2}\\(X', 'A\n1\nS\n1 (X')
    self.assertEqual(m.span(0, 1), ((0, 10), (5, 8)))
    m = regex.search('Derde\\s*:', 'aaaaaa:\nDerde:')
    self.assertEqual(m.span(), (8, 14))
    m = regex.search('Derde\\s*:', 'aaaaa:\nDerde:')
    self.assertEqual(m.span(), (7, 13))