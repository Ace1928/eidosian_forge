import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_matches_stdlib(self):
    """Should match the stdlib literal_eval if no names or indexing"""
    self.assertMatchesStdlib('[1]')
    self.assertMatchesStdlib('{(1,): [2,3,{}]}')
    self.assertMatchesStdlib('{1, 2}')