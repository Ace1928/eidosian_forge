from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
def test_inline_non_assign_fails(self):
    src = 'CONSTANT1, CONSTANT2 = values'
    t = ast.parse(src)
    with self.assertRaisesRegexp(inline.InlineError, "'CONSTANT1' is not declared in an assignment"):
        inline.inline_name(t, 'CONSTANT1')