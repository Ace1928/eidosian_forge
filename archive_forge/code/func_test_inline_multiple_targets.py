from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
def test_inline_multiple_targets(self):
    src = 'x = y = z = 1\na = x + y\n'
    t = ast.parse(src)
    inline.inline_name(t, 'y')
    self.checkAstsEqual(t, ast.parse('x = z = 1\na = x + 1\n'))