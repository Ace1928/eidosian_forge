from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_ignore_case(self):
    self.assertEqual(regex.match('abc', 'ABC', regex.I)[0], 'ABC')
    self.assertEqual(regex.match(b'abc', b'ABC', regex.I)[0], b'ABC')
    self.assertEqual(regex.match('(a\\s[^a]*)', 'a bb', regex.I)[1], 'a bb')
    self.assertEqual(regex.match('(a\\s[abc])', 'a b', regex.I)[1], 'a b')
    self.assertEqual(regex.match('(a\\s[abc]*)', 'a bb', regex.I)[1], 'a bb')
    self.assertEqual(regex.match('((a)\\s\\2)', 'a a', regex.I)[1], 'a a')
    self.assertEqual(regex.match('((a)\\s\\2*)', 'a aa', regex.I)[1], 'a aa')
    self.assertEqual(regex.match('((a)\\s(abc|a))', 'a a', regex.I)[1], 'a a')
    self.assertEqual(regex.match('((a)\\s(abc|a)*)', 'a aa', regex.I)[1], 'a aa')
    self.assertEqual(regex.match('[Z-a]', '_').span(), (0, 1))
    self.assertEqual(regex.match('(?i)[Z-a]', '_').span(), (0, 1))
    self.assertEqual(bool(regex.match('(?i)nao', 'nAo')), True)
    self.assertEqual(bool(regex.match('(?i)n\\xE3o', 'nÃo')), True)
    self.assertEqual(bool(regex.match('(?i)n\\xE3o', 'NÃO')), True)
    self.assertEqual(bool(regex.match('(?i)s', 'ſ')), True)