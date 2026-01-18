from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
def test_inline_conditional_fails(self):
    src = 'if define:\n  x = 1\na = x\n'
    t = ast.parse(src)
    with self.assertRaisesRegexp(inline.InlineError, "'x' is not a top-level name"):
        inline.inline_name(t, 'x')