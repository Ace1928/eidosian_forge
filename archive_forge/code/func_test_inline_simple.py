from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
def test_inline_simple(self):
    src = 'x = 1\na = x\n'
    t = ast.parse(src)
    inline.inline_name(t, 'x')
    self.checkAstsEqual(t, ast.parse('a = 1\n'))