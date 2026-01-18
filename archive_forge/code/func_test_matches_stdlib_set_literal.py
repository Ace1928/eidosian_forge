import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
@unittest.skipUnless(sys.version_info[:2] >= (3, 9), 'Only Python3.9 evaluates set()')
def test_matches_stdlib_set_literal(self):
    """set() is evaluated"""
    self.assertMatchesStdlib('set()')