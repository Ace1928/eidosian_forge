import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def assertCannotEval(self, line, ns=None):
    assert line.count('|') == 1
    cursor_offset = line.find('|')
    line = line.replace('|', '')
    with self.assertRaises(EvaluationError):
        evaluate_current_expression(cursor_offset, line, ns)