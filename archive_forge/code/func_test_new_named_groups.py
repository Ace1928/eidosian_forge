from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_new_named_groups(self):
    m0 = regex.match('(?P<a>\\w)', 'x')
    m1 = regex.match('(?<a>\\w)', 'x')
    if not (m0 and m1 and (m0[:] == m1[:])):
        self.fail('Failed')