import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def assertEvaled(self, line, value, ns=None):
    assert line.count('|') == 1
    cursor_offset = line.find('|')
    line = line.replace('|', '')
    self.assertEqual(evaluate_current_expression(cursor_offset, line, ns), value)