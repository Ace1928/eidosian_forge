import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def assertMatchesStdlib(self, expr):
    self.assertEqual(ast.literal_eval(expr), simple_eval(expr))