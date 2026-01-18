from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_atomic(self):
    self.assertEqual(regex.search('(?>a*)a', 'aa'), None)