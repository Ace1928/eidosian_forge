from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_groupdict(self):
    self.assertEqual(regex.match('(?P<first>first) (?P<second>second)', 'first second').groupdict(), {'first': 'first', 'second': 'second'})