import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_function_calls_raise(self):
    with self.assertRaises(ValueError):
        simple_eval('1()')