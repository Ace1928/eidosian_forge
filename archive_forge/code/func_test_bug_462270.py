from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_bug_462270(self):
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.sub('(?V0)x*', '-', 'abxd'), '-a-b--d-')
    else:
        self.assertEqual(regex.sub('(?V0)x*', '-', 'abxd'), '-a-b-d-')
    self.assertEqual(regex.sub('(?V1)x*', '-', 'abxd'), '-a-b--d-')
    self.assertEqual(regex.sub('x+', '-', 'abxd'), 'ab-d')