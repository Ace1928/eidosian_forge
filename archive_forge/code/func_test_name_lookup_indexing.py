import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_name_lookup_indexing(self):
    """Names can be looked up in a namespace"""
    self.assertEqual(simple_eval('a[b]', {'a': {'c': 1}, 'b': 'c'}), 1)