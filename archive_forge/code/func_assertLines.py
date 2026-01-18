from __future__ import absolute_import
import os
import re
import unittest
import shlex
import sys
import tempfile
import textwrap
from io import open
from functools import partial
from .Compiler import Errors
from .CodeWriter import CodeWriter
from .Compiler.TreeFragment import TreeFragment, strip_common_indent
from .Compiler.Visitor import TreeVisitor, VisitorTransform
from .Compiler import TreePath
def assertLines(self, expected, result):
    """Checks that the given strings or lists of strings are equal line by line"""
    if not isinstance(expected, list):
        expected = expected.split(u'\n')
    if not isinstance(result, list):
        result = result.split(u'\n')
    for idx, (expected_line, result_line) in enumerate(zip(expected, result)):
        self.assertEqual(expected_line, result_line, 'Line %d:\nExp: %s\nGot: %s' % (idx, expected_line, result_line))
    self.assertEqual(len(expected), len(result), 'Unmatched lines. Got:\n%s\nExpected:\n%s' % ('\n'.join(expected), u'\n'.join(result)))